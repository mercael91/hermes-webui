"""A suppressed Agent relaunch activates its dependencies inside this process.

When the process owner suppressed the relaunch (``HERMES_DISABLE_LAZY_INSTALLS``
set to a value hermes-agent reads as the opt-out), importing ``hermes_bootstrap``
activates the Agent's dependency layer *here*: ``harden_import_path`` re-fronts
the checkout and ``pm.activate_dependencies`` front-loads the venv selected for
the pending install -- another Python release's when a lazy install is pending --
while rewriting ``PYTHONPATH``/``PATH``/``VIRTUAL_ENV``. Left in place, that venv
resolves the Agent's C-extensions inside an interpreter that cannot load them.

The boundary puts the owner's environment back, drops only the entries that
belong to another release, keeps the venv the installed Agent runs from ahead of
the server's own paths, and leaves the Agent's environ-array wrappers alone --
those are what keep a concurrent read from touching a freed array.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import sysconfig

import pytest

from managed_agent_startup import LAZY_INSTALL_GUARD, agent_import_boundary

#: The Agent's launch layer, as ``pm.activate_dependencies`` leaves the process:
#: the checkout hardened onto ``sys.path``, the selected venv front-loaded, and
#: the environ helpers taken over (``install_never_free_environ``).
_STUB = """\
import os
import sys

sys.path.insert(0, {venv!r})
os.environ["PYTHONPATH"] = {agent!r}
os.environ.pop("VIRTUAL_ENV", None)
os.environ["PATH"] = {venv!r} + os.pathsep + os.environ.get("PATH", "")
os.putenv = lambda *args: None
os.unsetenv = lambda *args: None
"""

#: Stands in for ``pydantic_core``: a package whose ``__init__`` imports the
#: compiled part next to it.
_PACKAGE = "agent_generation_dep"


def _other_release() -> str:
    return f"{sys.version_info[0]}.{sys.version_info[1] + 1}"


def _agent_with_venv(tmp_path, monkeypatch, release, *, cfg_key="version_info"):
    """A fake Agent checkout whose launch layer activates a ``release`` venv.

    ``cfg_key`` selects how the venv names its generation: ``venv`` writes
    ``version``, uv writes ``version_info``.
    """
    agent = tmp_path / "agent"
    agent.mkdir()
    venv_root = tmp_path / "agent-venv"
    venv = venv_root / "lib" / f"python{release}" / "site-packages"
    venv.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text(
        f"{cfg_key} = {release}.1\n", encoding="utf-8"
    )
    (agent / "hermes_bootstrap.py").write_text(
        _STUB.format(venv=str(venv), agent=str(agent)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(agent))
    # The owner of the process suppressed the relaunch: the precondition for the
    # in-process activation these tests exercise.
    monkeypatch.setenv(LAZY_INSTALL_GUARD, "1")
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.delitem(sys.modules, "hermes_bootstrap", raising=False)
    return agent, venv


@pytest.mark.parametrize("cfg_key", ["version", "version_info"])
def test_other_release_venv_is_dropped_and_environment_restored(
    tmp_path, monkeypatch, cfg_key
):
    """#7982: a pending lazy install activates a venv this interpreter cannot load."""
    agent, venv = _agent_with_venv(
        tmp_path, monkeypatch, _other_release(), cfg_key=cfg_key
    )
    host_sys_path = sys.path[:]
    host_path = os.environ["PATH"]
    host_putenv, host_unsetenv = os.putenv, os.unsetenv

    try:
        with agent_import_boundary():
            importlib.import_module("hermes_bootstrap")
            assert str(venv) in sys.path
            assert os.environ["PYTHONPATH"] == str(agent)
            assert "VIRTUAL_ENV" not in os.environ
    finally:
        sys.modules.pop("hermes_bootstrap", None)

    assert sys.path == host_sys_path
    assert os.environ["PYTHONPATH"] == "/host/site-packages"
    assert os.environ["VIRTUAL_ENV"] == "/host/venv"
    assert os.environ["PATH"] == host_path

    # The Agent took over the environ helpers to keep a freed environ array from
    # being read; the boundary leaves that protection in place, so put back what
    # the stub replaced for the rest of the suite.
    assert (os.putenv, os.unsetenv) != (host_putenv, host_unsetenv)
    os.putenv, os.unsetenv = host_putenv, host_unsetenv


def test_in_process_agent_venv_stays_importable(tmp_path, monkeypatch):
    """The installed Agent's own venv is this process's dependency source."""
    release = f"{sys.version_info[0]}.{sys.version_info[1]}"
    _, venv = _agent_with_venv(tmp_path, monkeypatch, release)
    host_sys_path = sys.path[:]

    try:
        with agent_import_boundary():
            importlib.import_module("hermes_bootstrap")
    finally:
        sys.modules.pop("hermes_bootstrap", None)

    assert str(venv) in sys.path
    # ``pm.activate_dependencies`` front-loads the selected venv, so the Agent
    # keeps resolving its dependencies ahead of the server's own paths.
    assert sys.path == [str(venv)] + host_sys_path
    assert os.environ["PYTHONPATH"] == "/host/site-packages"
    assert os.environ["VIRTUAL_ENV"] == "/host/venv"


def _snapshot():
    """This process's interpreter state, as the boundary must hand it back."""
    return (
        sys.path[:],
        os.environ.get("PATH"),
        os.environ.get("PYTHONPATH"),
        os.environ.get("VIRTUAL_ENV"),
    )


def _assert_restored(snapshot):
    path, host_path, pythonpath, virtual_env = snapshot
    assert sys.path == path
    assert os.environ.get("PATH") == host_path
    assert os.environ.get("PYTHONPATH") == pythonpath
    assert os.environ.get("VIRTUAL_ENV") == virtual_env


def test_malformed_pyvenv_cfg_is_tolerated(tmp_path, monkeypatch):
    """A ``version`` without a minor part must not escape as IndexError.

    The release parser runs while the boundary rebuilds ``sys.path``, so an
    exception there would skip the rest of the repair.
    """
    _agent_with_venv(tmp_path, monkeypatch, _other_release())
    (tmp_path / "agent-venv" / "pyvenv.cfg").write_text(
        "version_info = 3\n", encoding="utf-8"
    )
    snapshot = _snapshot()

    try:
        with agent_import_boundary():
            importlib.import_module("hermes_bootstrap")
    finally:
        sys.modules.pop("hermes_bootstrap", None)

    _assert_restored(snapshot)


def test_unreadable_pyvenv_cfg_is_tolerated(tmp_path, monkeypatch):
    """An unreadable path must not escape as OSError either."""
    _agent_with_venv(tmp_path, monkeypatch, _other_release())
    real_is_file = Path.is_file

    def unreadable(self):
        if self.name == "pyvenv.cfg":
            raise PermissionError(13, "Permission denied")
        return real_is_file(self)

    snapshot = _snapshot()

    try:
        monkeypatch.setattr(Path, "is_file", unreadable)
        with agent_import_boundary():
            importlib.import_module("hermes_bootstrap")
    finally:
        sys.modules.pop("hermes_bootstrap", None)

    _assert_restored(snapshot)


def _foreign_artifact_release() -> tuple[str, str]:
    """The other release, and the extension suffix that names it."""
    release = _other_release()
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    running = f"{sys.version_info[0]}{sys.version_info[1]}"
    other = release.replace(".", "")
    if running not in suffix:
        pytest.skip(f"no extension suffix naming {running} on this platform")
    return release, suffix.replace(running, other)


def _foreign_package(tmp_path, monkeypatch) -> Path:
    """A package whose only compiled part belongs to another Python release.

    ``pydantic_core`` in production: its ``__init__`` imports the compiled
    ``_pydantic_core`` beside it, and the copy the foreign venv ships is not
    loadable by this interpreter, so the import fails with ``No module named
    'pydantic_core._pydantic_core'`` -- a real import failure, not a stand-in.
    """
    release, tag = _foreign_artifact_release()
    _, venv = _agent_with_venv(tmp_path, monkeypatch, release)
    foreign = venv / _PACKAGE
    foreign.mkdir()
    (foreign / f"_{_PACKAGE}{tag}").write_bytes(b"")
    (foreign / "__init__.py").write_text(
        f"from ._{_PACKAGE} import core\n", encoding="utf-8"
    )
    host = tmp_path / "host" / _PACKAGE
    host.mkdir(parents=True)
    (host / "__init__.py").write_text("core = 'host'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(host.parent))
    return venv


def test_a_foreign_artifact_breaks_the_import_without_the_boundary(
    tmp_path, monkeypatch
):
    """The shape the owner asked for, unguarded: this is the #7982 failure.

    Nothing here stands in for the failure -- the import system refuses the
    artifact the front-loaded venv ships, exactly as it refuses another
    release's ``_pydantic_core``.
    """
    _foreign_package(tmp_path, monkeypatch)
    host_sys_path = sys.path[:]
    try:
        importlib.import_module("hermes_bootstrap")
        with pytest.raises(ModuleNotFoundError) as failure:
            importlib.import_module(_PACKAGE)
        assert failure.value.name == f"{_PACKAGE}._{_PACKAGE}"
    finally:
        sys.modules.pop("hermes_bootstrap", None)
        sys.modules.pop(_PACKAGE, None)
        sys.path[:] = host_sys_path


def test_the_boundary_keeps_the_host_copy_importable(tmp_path, monkeypatch):
    """With the boundary the same import resolves what this interpreter owns."""
    _foreign_package(tmp_path, monkeypatch)
    try:
        with agent_import_boundary():
            importlib.import_module("hermes_bootstrap")
        assert importlib.import_module(_PACKAGE).core == "host"
    finally:
        sys.modules.pop("hermes_bootstrap", None)
        sys.modules.pop(_PACKAGE, None)
