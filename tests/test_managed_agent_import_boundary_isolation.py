"""The Agent import boundary keeps this process's interpreter self-consistent.

``hermes_bootstrap``'s module body re-fronts its checkout on ``sys.path``
(``harden_import_path``) and ``pm.activate_dependencies`` front-loads the Agent's
selected venv -- its own Python when a lazy install is pending -- while rewriting
``PYTHONPATH``/``PATH``/``VIRTUAL_ENV`` and taking over ``os.putenv``. A venv of
another Python release left in place makes the next import resolve the Agent's
C-extensions (``pydantic_core._pydantic_core``) inside an interpreter that cannot
load them; the venv the installed Agent actually runs from stays importable, and
the server's own environment comes back untouched either way.
"""

from __future__ import annotations

import importlib
import os
import sys

from managed_agent_startup import agent_import_boundary

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


def _agent_with_venv(tmp_path, monkeypatch, release):
    """A fake Agent checkout whose bootstrap activates a ``release`` venv."""
    agent = tmp_path / "agent"
    agent.mkdir()
    venv_root = tmp_path / "agent-venv"
    venv = venv_root / "lib" / f"python{release}" / "site-packages"
    venv.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text(f"version = {release}.1\n", encoding="utf-8")
    (agent / "hermes_bootstrap.py").write_text(
        _STUB.format(venv=str(venv), agent=str(agent)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(agent))
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.delitem(sys.modules, "hermes_bootstrap", raising=False)
    return agent, venv


def test_other_release_venv_is_dropped_and_environment_restored(tmp_path, monkeypatch):
    """#7982: a pending lazy install activates a venv this interpreter cannot load."""
    release = f"{sys.version_info[0]}.{sys.version_info[1] + 1}"
    agent, venv = _agent_with_venv(tmp_path, monkeypatch, release)
    host_sys_path = sys.path[:]
    host_path = os.environ["PATH"]
    host_putenv, host_unsetenv = os.putenv, os.unsetenv

    try:
        with agent_import_boundary():
            importlib.import_module("hermes_bootstrap")
            assert str(venv) in sys.path
            assert os.environ["PYTHONPATH"] == str(agent)
            assert "VIRTUAL_ENV" not in os.environ
            assert os.putenv is not host_putenv
    finally:
        # The fake launch layer must not stay cached for later tests.
        sys.modules.pop("hermes_bootstrap", None)

    assert sys.path == host_sys_path
    assert os.environ["PYTHONPATH"] == "/host/site-packages"
    assert os.environ["VIRTUAL_ENV"] == "/host/venv"
    assert os.environ["PATH"] == host_path
    assert (os.putenv, os.unsetenv) == (host_putenv, host_unsetenv)


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
    assert sys.path == host_sys_path + [str(venv)]
    assert os.environ["PYTHONPATH"] == "/host/site-packages"
    assert os.environ["VIRTUAL_ENV"] == "/host/venv"
