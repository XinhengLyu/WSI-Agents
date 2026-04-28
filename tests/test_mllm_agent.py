"""Tests for MLLM_agent helpers and constructors."""
import json
from unittest.mock import MagicMock

import pytest

from JsonlResponseReader import JsonlResponseReader
from MLLM_agent import _load_responses, MorphologyAgent, DiagnosisAgent, TreatmentAgent, ReportAgent
from config import Config


@pytest.fixture
def reader_with_three_models(tmp_path):
    paths = {}
    for key in ("mllm_1", "mllm_2", "mllm_3"):
        f = tmp_path / f"{key}.jsonl"
        f.write_text(json.dumps({"question_id": "q1", "text": f"answer-from-{key}"}) + "\n")
        paths[key] = str(f)
    return JsonlResponseReader(paths)


def test_load_responses_returns_three_in_order(reader_with_three_models, monkeypatch):
    monkeypatch.setattr(Config, "TASK_MLLM_KEYS", {
        "Morphology": ["mllm_1", "mllm_2", "mllm_3"],
    })
    r1, r2, r3 = _load_responses(reader_with_three_models, "Morphology", "q1")
    assert r1 == "answer-from-mllm_1"
    assert r2 == "answer-from-mllm_2"
    assert r3 == "answer-from-mllm_3"


def test_load_responses_raises_on_missing(reader_with_three_models, monkeypatch):
    monkeypatch.setattr(Config, "TASK_MLLM_KEYS", {
        "Morphology": ["mllm_1", "mllm_2", "mllm_3"],
    })
    with pytest.raises(ValueError, match="Missing responses"):
        _load_responses(reader_with_three_models, "Morphology", "q_missing")


@pytest.mark.parametrize("agent_cls", [MorphologyAgent, DiagnosisAgent, TreatmentAgent, ReportAgent])
def test_mllm_agents_construct(agent_cls, agent_ctx):
    with agent_ctx():
        agent = agent_cls(model_client=MagicMock(), response_reader=MagicMock())
    assert agent._model_client is not None
    assert agent._response_reader is not None
