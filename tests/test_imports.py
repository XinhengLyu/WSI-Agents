"""Smoke test: every project module must import cleanly."""
import importlib
import pytest

MODULES = [
    "config",
    "base_models",
    "ScoreCalculator",
    "JsonlResponseReader",
    "ClassifierResponseReader",
    "model_client",
    "knowledge_base",
    "agent",
    "MLLM_agent",
    "InternalValidation",
    "ExternalValidation",
    "IntegrationAgent",
    "MedicalAnalysisSystem",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    importlib.import_module(module_name)
