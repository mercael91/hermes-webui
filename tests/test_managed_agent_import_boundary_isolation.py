"""The Agent import boundary must not leak the Agent's import path into this process.

hermes_bootstrap's module body re-fronts its checkout on ``sys.path``
(``harden_import_path``) and ``pm.activate_dependencies`` front-loads the Agent's
selected venv -- its own Python when a lazy install is pending -- while rewriting
``PYTHONPATH``/``PATH``/``VIRTUAL_ENV`` and taking over ``os.putenv``. Left in
place in a long-lived server, the next import resolves the Agent's C-extensions
(``pydantic_core._pydantic_core``) inside an interpreter that cannot load them.
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


def test_boundary_restores_host_import_path_and_environment(tmp_path, monkeypatch):
    agent = tmp_path / "agent"
    venv = tmp_path / "agent-venv" / "site-packages"
    venv.mkdir(parents=True)
    agent.mkdir()
    (agent / "hermes_bootstrap.py").write_text(
        _STUB.format(venv=str(venv), agent=str(agent)), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(agent))
    monkeypatch.setenv("VIRTUAL_ENV", "/host/venv")
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.delitem(sys.modules, "hermes_bootstrap", raising=False)
    host_sys_path = sys.path[:]
    host_path = os.environ["PATH"]
    host_putenv, host_unsetenv = os.putenv, os.unsetenv

    with agent_import_boundary():
        importlib.import_module("hermes_bootstrap")
        assert str(venv) in sys.path
        assert os.environ["PYTHONPATH"] == str(agent)
        assert "VIRTUAL_ENV" not in os.environ
        assert os.putenv is not host_putenv

    assert sys.path == host_sys_path
    assert os.environ["PYTHONPATH"] == "/host/site-packages"
    assert os.environ["VIRTUAL_ENV"] == "/host/venv"
    assert os.environ["PATH"] == host_path
    assert (os.putenv, os.unsetenv) == (host_putenv, host_unsetenv)
