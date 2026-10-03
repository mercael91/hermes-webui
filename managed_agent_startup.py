"""Initialize a discovered Agent source checkout before importing WebUI modules."""

import contextlib
import importlib
import os
from pathlib import Path
import sys


#: hermes-agent's internal bridge variable (hermes_bootstrap -> venv_sync,
#: pm.install). hermes_bootstrap.py relaunches the process via os.execv into
#: its managed sandbox the moment it is imported with a pending lazy
#: install/update; pm.install.lazy_installs_allowed() also reads it as an
#: unconditional override of security.allow_lazy_installs. The WebUI needs the
#: relaunch disabled only while Agent code is being imported, so this guard is
#: scoped to that boundary instead of the process lifetime.
LAZY_INSTALL_GUARD = "HERMES_DISABLE_LAZY_INSTALLS"


@contextlib.contextmanager
def agent_import_boundary():
    """Disable the Agent's lazy-install interception for exactly one import.

    The sandbox hermes_bootstrap relaunches into carries no WebUI
    dependencies, so an unguarded Agent import replaces this process with one
    that dies on ``import yaml`` and the service restarts forever. Keeping the
    override for the whole process would instead outlive startup and silently
    disable on-demand installs the operator allowed through
    ``security.allow_lazy_installs`` -- pm/install.py treats any truthy value
    as the policy. Whatever the operator had (set or unset) is restored as
    soon as the import returns, including when it fails.
    """
    previous = os.environ.get(LAZY_INSTALL_GUARD)
    os.environ[LAZY_INSTALL_GUARD] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LAZY_INSTALL_GUARD, None)
        else:
            os.environ[LAZY_INSTALL_GUARD] = previous


def activate_managed_agent() -> None:
    agent_dir = os.environ.get("HERMES_WEBUI_AGENT_DIR")
    if not agent_dir:
        return
    # Browser-only setups may deliberately point at an empty Agent directory.
    if not (Path(agent_dir) / "run_agent.py").is_file():
        return

    webui_root = str(Path(__file__).resolve().parent)
    if webui_root not in sys.path:
        sys.path.insert(0, webui_root)
    # Activate dependencies without importing the application: api.config must
    # select the active profile before Agent modules cache profile-sensitive paths.
    # Older Agents and browser-only shims have no bootstrap layer to activate; for
    # them leave sys.path alone so api.config appends the Agent dir at the END as
    # before (a front position lets `pip install -t .` packages in the checkout
    # shadow site-packages).
    if (Path(agent_dir) / "hermes_bootstrap.py").is_file():
        # Bootstrap's probe adds the checkout to its own PYTHONPATH, not ours.
        if agent_dir not in sys.path:
            sys.path.insert(1, agent_dir)
        try:
            # Boundary: hermes_bootstrap's module-level launch layer would
            # otherwise replace this WebUI process with its sandbox.
            with agent_import_boundary():
                importlib.import_module("hermes_bootstrap")
        except Exception as exc:  # noqa: BLE001 - SystemExit (relaunch/repair exit) still propagates
            # A broken Agent must not stop WebUI from starting: before this hook the
            # Agent import was lazy and an ImportError only disabled chat, leaving the
            # UI, diagnostics and updater reachable. Keep that behavior.
            print(
                f"[!!] Hermes Agent dependency activation failed: {type(exc).__name__}: {exc}; "
                "continuing startup. If WebUI then fails to import a dependency, run "
                "`hermes pm repair` or set HERMES_WEBUI_PYTHON.",
                file=sys.stderr,
                flush=True,
            )
