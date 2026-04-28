"""Tests for InternalConsistencyAgent helpers."""
from unittest.mock import MagicMock

import pytest

from InternalValidation import InternalConsistencyAgent


@pytest.fixture
def agent(agent_ctx):
    with agent_ctx():
        return InternalConsistencyAgent(model_client=MagicMock())


def _model_dict(content, reasoning, issues=None):
    return {
        "content_score": content,
        "reasoning_score": reasoning,
        "issues_found": issues if issues is not None else [],
    }


def test_process_consistency_analysis_basic(agent):
    raw = {
        "mllm1_analysis": _model_dict(0.9, 0.7, ["overlap"]),
        "mllm2_analysis": _model_dict(0.5, 0.5),
        "mllm3_analysis": _model_dict(1.0, 1.0),
    }
    result = agent._process_consistency_analysis(raw)

    assert result.mllm1_analysis.content_score == 0.9
    assert result.mllm1_analysis.reasoning_score == 0.7
    assert result.mllm1_analysis.consistency_score == pytest.approx(0.8)
    assert result.mllm1_analysis.issues_found == ["overlap"]

    assert result.mllm2_analysis.consistency_score == pytest.approx(0.5)
    assert result.mllm3_analysis.consistency_score == pytest.approx(1.0)


def test_process_consistency_analysis_missing_keys_raises(agent):
    raw = {
        "mllm1_analysis": {"content_score": 0.5},  # missing reasoning_score, issues_found
        "mllm2_analysis": _model_dict(0.5, 0.5),
        "mllm3_analysis": _model_dict(0.5, 0.5),
    }
    with pytest.raises(ValueError, match="Missing required fields"):
        agent._process_consistency_analysis(raw)


def test_process_consistency_analysis_non_list_issues_coerced_to_empty(agent):
    raw = {
        "mllm1_analysis": {
            "content_score": 0.5, "reasoning_score": 0.5,
            "issues_found": "not a list",
        },
        "mllm2_analysis": _model_dict(0.5, 0.5),
        "mllm3_analysis": _model_dict(0.5, 0.5),
    }
    result = agent._process_consistency_analysis(raw)
    assert result.mllm1_analysis.issues_found == []


def test_process_consistency_analysis_string_scores_coerced(agent):
    """Defensively cast str → float (LLMs sometimes emit "0.7")."""
    raw = {
        "mllm1_analysis": {"content_score": "0.7", "reasoning_score": "0.3", "issues_found": []},
        "mllm2_analysis": _model_dict(0.5, 0.5),
        "mllm3_analysis": _model_dict(0.5, 0.5),
    }
    result = agent._process_consistency_analysis(raw)
    assert result.mllm1_analysis.content_score == pytest.approx(0.7)
    assert result.mllm1_analysis.consistency_score == pytest.approx(0.5)
