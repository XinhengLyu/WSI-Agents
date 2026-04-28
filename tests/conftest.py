"""Test bootstrap: project import path + helpers for autogen agent instantiation."""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def agent_ctx():
    """Context manager that lets `RoutedAgent` subclasses be instantiated outside of a runtime.

    Usage:
        def test_something(agent_ctx):
            with agent_ctx():
                agent = MyAgent(...)
    """
    from autogen_core import AgentId, SingleThreadedAgentRuntime
    from autogen_core._agent_instantiation import AgentInstantiationContext

    runtime = SingleThreadedAgentRuntime()
    agent_id = AgentId(type="test_agent", key="test")

    def _enter():
        return AgentInstantiationContext.populate_context((runtime, agent_id))

    return _enter


@pytest.fixture(autouse=True)
def _strip_socks_proxy_env(monkeypatch):
    """Local dev sometimes sets a SOCKS5H proxy that httpx can't parse — clear it."""
    for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                "HTTPS_PROXY", "https_proxy"):
        val = os.environ.get(var, "")
        if val.startswith("socks5h://"):
            monkeypatch.delenv(var, raising=False)
