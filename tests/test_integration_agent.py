"""Tests for IntegrationAgent helpers."""
import json
import types
from unittest.mock import MagicMock

import pytest

from IntegrationAgent import IntegrationAgent, _parse_review_response
from base_models import ModelReliability


@pytest.fixture
def agent(agent_ctx):
    """Build an IntegrationAgent with mocked clients (no LLM calls)."""
    clients = {
        "gpt-4o":   MagicMock(),
        "Claude":   MagicMock(),
        "Gemini":   MagicMock(),
        "DeepSeek": MagicMock(),
    }
    with agent_ctx():
        return IntegrationAgent(clients)


# ─────────────────── _parse_review_response ────────────────────

def test_parse_review_response_valid_json():
    resp = types.SimpleNamespace(content=json.dumps(
        {"verdict": "approve", "reasoning": "ok", "suggestion": ""}
    ))
    out = _parse_review_response(resp, "Claude")
    assert out["verdict"] == "approve"
    assert out["reasoning"] == "ok"


def test_parse_review_response_synonyms_normalize_to_approve():
    for synonym in ("accept", "yes", "agreed", "APPROVE"):
        resp = types.SimpleNamespace(content=json.dumps(
            {"verdict": synonym, "reasoning": "ok"}
        ))
        out = _parse_review_response(resp, "Claude")
        assert out["verdict"] == "approve", f"failed for synonym: {synonym}"


def test_parse_review_response_unknown_verdict_falls_back_to_reject():
    resp = types.SimpleNamespace(content=json.dumps(
        {"verdict": "maybe", "reasoning": "?"}
    ))
    out = _parse_review_response(resp, "Claude")
    assert out["verdict"] == "reject"


def test_parse_review_response_embedded_json():
    resp = types.SimpleNamespace(content=(
        'Some preamble text here.\n'
        '{"verdict": "approve", "reasoning": "looks fine"}\n'
        'and trailing text.'
    ))
    out = _parse_review_response(resp, "Claude")
    assert out["verdict"] == "approve"


def test_parse_review_response_no_content_attr():
    out = _parse_review_response(object(), "Claude")
    assert out["verdict"] == "reject"


def test_parse_review_response_no_json_at_all():
    resp = types.SimpleNamespace(content="totally not json")
    out = _parse_review_response(resp, "Claude")
    assert out["verdict"] == "reject"
    assert "totally not json" in out["reasoning"]


# ─────────────────── _select_best_response ────────────────────

def _r(score):
    return ModelReliability(
        knowledge_score=score, consistency_score=score,
        classifier_score=None, final_score=score,
        assessment_details="x",
    )


def test_select_best_response_picks_highest_final_score(agent):
    rels = {"mllm1": _r(0.4), "mllm2": _r(0.9), "mllm3": _r(0.7)}
    rel, name, score = agent._select_best_response(rels)
    assert name == "mllm2"
    assert score == 0.9
    assert rel.final_score == 0.9


# ───────────────── _create_enhancement_prompt ─────────────────

def test_create_enhancement_prompt_returns_empty_when_no_others(agent):
    prompt = agent._create_enhancement_prompt(
        refined_response="text",
        model_responses={"mllm1": "only one"},
        best_model="mllm1",
    )
    assert prompt == ""


def test_create_enhancement_prompt_excludes_best_and_falsy(agent):
    prompt = agent._create_enhancement_prompt(
        refined_response="best answer",
        model_responses={
            "mllm1": "best answer",
            "mllm2": "alt answer",
            "mllm3": "",  # falsy, should be excluded
        },
        best_model="mllm1",
    )
    assert "alt answer" in prompt
    assert prompt.count("===") >= 2  # one section header for mllm2


# ──────────────────── _try_process_results ─────────────────────

def test_try_process_results_waits_until_required_keys_arrive(agent, capsys):
    fake_consistency = MagicMock()
    agent._pending_results["sess1"] = {"consistency": fake_consistency}
    import asyncio
    asyncio.run(agent._try_process_results("sess1"))
    captured = capsys.readouterr()
    assert "Waiting" in captured.out


def test_try_process_results_waits_for_classifier_on_diagnosis(agent, capsys, monkeypatch):
    from config import Config
    monkeypatch.setattr(Config, "TASK_NAME", "Diagnosis")
    agent._pending_results["sess1"] = {
        "consistency":   MagicMock(),
        "verification":  MagicMock(),
        "analysis_task": MagicMock(),
    }
    import asyncio
    asyncio.run(agent._try_process_results("sess1"))
    captured = capsys.readouterr()
    assert "Waiting for classifier" in captured.out


def test_try_process_results_does_not_wait_for_classifier_on_treatment(agent, monkeypatch):
    """Treatment task: 3 keys is enough; processing should advance past the wait."""
    from config import Config
    from base_models import KnowledgeValidation, ModelAnalysis, VerificationResult, ConsistencyResult, AnalysisTask
    monkeypatch.setattr(Config, "TASK_NAME", "Treatment")

    kv = KnowledgeValidation(score=0.5, matched_rules=[], unmatched_rules=[], validation_details="")
    ma = ModelAnalysis(content_score=0.5, reasoning_score=0.5, consistency_score=0.5, issues_found=[])
    agent._pending_results["sess1"] = {
        "consistency":   ConsistencyResult(content="{}", mllm1_analysis=ma, mllm2_analysis=ma, mllm3_analysis=ma),
        "verification":  VerificationResult(content="{}", mllm1_validation=kv, mllm2_validation=kv, mllm3_validation=kv),
        "analysis_task": AnalysisTask(question_id="q1", question="?",
                                      mllm1_response="r1", mllm2_response="r2", mllm3_response="r3"),
    }

    def boom(*args, **kwargs):
        raise RuntimeError("got past the classifier wait")

    monkeypatch.setattr(agent, "_select_best_response", boom)

    import asyncio
    with pytest.raises(RuntimeError, match="got past the classifier wait"):
        asyncio.run(agent._try_process_results("sess1"))


def test_try_process_results_falls_back_when_classifier_predictions_empty(agent, monkeypatch):
    """Diagnosis task: classifier present but with empty predictions → 6/4 reweight path."""
    from config import Config
    from base_models import (
        ClassifierVerification, ClassifierVerificationResult, AnalysisTask,
        KnowledgeValidation, ModelAnalysis, VerificationResult, ConsistencyResult,
    )

    monkeypatch.setattr(Config, "TASK_NAME", "Diagnosis")

    empty_cv = ClassifierVerification(
        multi_modal_consistency=0.0,
        classifier_internal_consistency=0.0,
        classifier_verification_score=0.0,
        classifier_predictions=[],
    )
    fake_task = AnalysisTask(
        question_id="q1", question="?",
        mllm1_response="r1", mllm2_response="r2", mllm3_response="r3",
    )
    kv = KnowledgeValidation(score=0.5, matched_rules=[], unmatched_rules=[], validation_details="")
    ma = ModelAnalysis(content_score=0.5, reasoning_score=0.5, consistency_score=0.5, issues_found=[])
    agent._pending_results["sess1"] = {
        "consistency":   ConsistencyResult(content="{}", mllm1_analysis=ma, mllm2_analysis=ma, mllm3_analysis=ma),
        "verification":  VerificationResult(content="{}", mllm1_validation=kv, mllm2_validation=kv, mllm3_validation=kv),
        "analysis_task": fake_task,
        "classifier": ClassifierVerificationResult(
            mllm1_result=empty_cv, mllm2_result=empty_cv, mllm3_result=empty_cv,
            content="{}", original_task=fake_task,
        ),
    }

    captured_kwargs = {"calls": []}

    def fake_calculate(k_score, c_score, classifier_score):
        captured_kwargs["calls"].append(classifier_score)
        return 0.5

    monkeypatch.setattr(agent._calculator, "calculate_final_score", fake_calculate)

    def short_circuit(*args, **kwargs):
        raise RuntimeError("reached scoring")

    monkeypatch.setattr(agent, "_create_integration_prompt", short_circuit)

    import asyncio
    with pytest.raises(RuntimeError, match="reached scoring"):
        asyncio.run(agent._try_process_results("sess1"))

    # Each of the 3 reliability rows should have called the calculator with classifier_score=None
    assert captured_kwargs["calls"] == [None, None, None]


# ─────────────── PRIMARY_MODEL_KEY constant ────────────────

def test_primary_model_key_is_gpt4o(agent):
    assert IntegrationAgent.PRIMARY_MODEL_KEY == "gpt-4o"
    assert agent._model_client is agent._model_clients["gpt-4o"]
