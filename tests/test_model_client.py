"""Smoke tests for model_client factory."""
import os

from model_client import create_model_client, DEFAULT_MODEL_CAPABILITIES


def test_create_model_client_returns_client(monkeypatch):
    """Factory should construct a client without contacting the network."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = create_model_client("gpt-4o")
    assert client is not None


def test_default_capabilities_are_present():
    for key in ("vision", "function_calling", "json_output"):
        assert key in DEFAULT_MODEL_CAPABILITIES
        assert DEFAULT_MODEL_CAPABILITIES[key] is True


def test_create_model_client_with_custom_capabilities():
    custom = {"vision": False, "function_calling": True, "json_output": False}
    client = create_model_client("gpt-4o", model_capabilities=custom)
    assert client is not None
