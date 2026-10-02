"""Regression test for the lazy-install sandbox interception crash loop.

hermes-agent's hermes_bootstrap.py hijacks the process via os.execv when a
lazy install/update is pending, relaunching server.py inside an isolated
Python 3.14 sandbox that lacks Web UI deps (yaml), causing an infinite
systemd restart loop. bootstrap.py must set HERMES_DISABLE_LAZY_INSTALLS=1
after the repository .env is loaded and before any hermes-agent code can be
imported — the .env loader assigns unconditionally and would otherwise let a
`HERMES_DISABLE_LAZY_INSTALLS=0` entry re-enable the crash loop.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "bootstrap.py"


def _import_bootstrap_and_env(bootstrap_path: Path, cwd: Path, extra_env: dict) -> dict:
    """Import *bootstrap_path* in a subprocess and return the resulting env."""
    env = os.environ.copy()
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    env.update(extra_env)
    script = (
        "import importlib.util, json, os\n"
        f"spec = importlib.util.spec_from_file_location('bootstrap', {str(bootstrap_path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print(json.dumps({k: os.environ.get(k) for k in (\n"
        "    'HERMES_DISABLE_LAZY_INSTALLS', 'HERMES_WEBUI_DOTENV_CANARY')}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )
    assert result.returncode == 0, result.stderr
    import json

    return json.loads(result.stdout.strip().splitlines()[-1])


def test_bootstrap_sets_disable_lazy_installs() -> None:
    """Importing bootstrap.py must set HERMES_DISABLE_LAZY_INSTALLS=1."""
    seen = _import_bootstrap_and_env(BOOTSTRAP, REPO_ROOT, {})
    assert seen["HERMES_DISABLE_LAZY_INSTALLS"] == "1"


def test_dotenv_cannot_reenable_lazy_installs(tmp_path: Path) -> None:
    """A repo .env setting HERMES_DISABLE_LAZY_INSTALLS=0 must not win.

    The escape hatch is established after ``_load_repo_dotenv()``, so a dotenv
    entry cannot restore the lazy-install interception that crash-loops the
    Web UI. The canary proves the .env file itself was still loaded.
    """
    copied = tmp_path / "bootstrap.py"
    copied.write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "HERMES_DISABLE_LAZY_INSTALLS=0\nHERMES_WEBUI_DOTENV_CANARY=from-dotenv\n",
        encoding="utf-8",
    )

    seen = _import_bootstrap_and_env(copied, tmp_path, {})
    assert seen["HERMES_WEBUI_DOTENV_CANARY"] == "from-dotenv", seen
    assert seen["HERMES_DISABLE_LAZY_INSTALLS"] == "1", seen


def test_escape_hatch_set_before_hermes_imports() -> None:
    """The env var must be set at the top of bootstrap.py, before imports
    that could pull in hermes-agent code."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    marker = 'os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"'
    assert marker in source
    # The assignment must appear before the first `import` of a hermes-agent
    # module (run_agent / hermes_bootstrap) anywhere in the file.
    marker_pos = source.index(marker)
    for needle in ("import run_agent", "from run_agent", "import hermes_bootstrap"):
        if needle in source:
            assert source.index(needle) > marker_pos
    # And it must come after the repo .env load, so dotenv cannot undo it.
    assert source.index(marker) > source.index("_load_repo_dotenv()\n", source.index('def _load_repo_dotenv'))
