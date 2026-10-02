"""Regression test for the lazy-install sandbox interception crash loop.

hermes-agent's hermes_bootstrap.py hijacks the process via os.execv when a
lazy install/update is pending, relaunching server.py inside an isolated
Python 3.14 sandbox that lacks Web UI deps (yaml), causing an infinite
systemd restart loop. bootstrap.py must set HERMES_DISABLE_LAZY_INSTALLS=1
before any hermes-agent code can be imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "bootstrap.py"


def test_bootstrap_sets_disable_lazy_installs() -> None:
    """Importing bootstrap.py must set HERMES_DISABLE_LAZY_INSTALLS=1."""
    env = os.environ.copy()
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    # Import bootstrap in a subprocess so we observe the module-level side
    # effect without polluting this test process's environment.
    script = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('bootstrap', {str(BOOTSTRAP)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print(os.environ.get('HERMES_DISABLE_LAZY_INSTALLS', ''))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"


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
