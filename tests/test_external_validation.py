"""Tests for ExternalValidation agents (pure helpers)."""
from unittest.mock import MagicMock

import pytest

from ExternalValidation import KnowledgeVerificationAgent, ClassifierVerificationAgent
from base_models import ClassifierPrediction


# ─────────────── KnowledgeVerificationAgent._process_verification ────────────────

@pytest.fixture
def kv_agent(agent_ctx):
    with agent_ctx():
        return KnowledgeVerificationAgent(model_client=MagicMock(), knowledge_base=MagicMock())


def _v(matched, unmatched, incorrect):
    return {
        "matched_rules": matched,
        "unmatched_rules": unmatched,
        "incorrect_findings": incorrect,
        "validation_details": "details",
    }


def test_verification_score_basic(kv_agent):
    raw = {
        "mllm1_validation": _v(["a", "b", "c"], ["d"], []),
        "mllm2_validation": _v([], ["a", "b"], []),
        "mllm3_validation": _v(["a", "b"], [], []),
    }
    result = kv_agent._process_verification(raw)
    assert result.mllm1_validation.score == pytest.approx(0.75)  # 3 / (3+1)
    assert result.mllm2_validation.score == pytest.approx(0.0)   # 0 / 2
    assert result.mllm3_validation.score == pytest.approx(1.0)   # 2 / 2


def test_verification_score_deducts_for_incorrect_findings(kv_agent):
    raw = {
        "mllm1_validation": _v(["a", "b", "c", "d"], [], ["x", "x"]),  # 1.0 - 0.2 = 0.8
        "mllm2_validation": _v(["a"], [],  []),
        "mllm3_validation": _v(["a"], [],  []),
    }
    result = kv_agent._process_verification(raw)
    assert result.mllm1_validation.score == pytest.approx(0.8)


def test_verification_score_clamps_to_zero(kv_agent):
    """Many incorrect findings should not drive the score negative."""
    raw = {
        "mllm1_validation": _v(["a"], [], ["x"] * 100),
        "mllm2_validation": _v(["a"], [], []),
        "mllm3_validation": _v(["a"], [], []),
    }
    result = kv_agent._process_verification(raw)
    assert result.mllm1_validation.score == 0.0


def test_verification_score_no_criteria_returns_zero(kv_agent):
    raw = {
        "mllm1_validation": _v([], [], []),
        "mllm2_validation": _v(["a"], [], []),
        "mllm3_validation": _v(["a"], [], []),
    }
    result = kv_agent._process_verification(raw)
    assert result.mllm1_validation.score == 0.0


# ───────── ClassifierVerificationAgent._calculate_verification_scores ─────────

@pytest.fixture
def cv_agent(agent_ctx):
    with agent_ctx():
        return ClassifierVerificationAgent(
            model_client=MagicMock(),
            classifier_reader=MagicMock(),
            classifier_models=["Conch", "TITAN"],
        )


def test_classifier_scores_with_no_predictions(cv_agent):
    import asyncio
    cv = asyncio.run(cv_agent._calculate_verification_scores(
        diagnosis={"diagnosis": "LUAD"},
        classifier_predictions=[],
    ))
    assert cv.classifier_predictions == []
    assert cv.multi_modal_consistency == 0.0
    assert cv.classifier_internal_consistency == 0.0
    assert cv.classifier_verification_score == 0.0


def test_classifier_internal_consistency_unanimous(cv_agent, monkeypatch):
    # Stub out compare_diagnoses so we don't hit an LLM
    async def fake_compare(self, a, b):
        return 1.0 if a.lower() == b.lower() else 0.0
    monkeypatch.setattr(ClassifierVerificationAgent, "compare_diagnoses", fake_compare)

    preds = [
        ClassifierPrediction(model_name="Conch", label="LUAD", confidence=0.9),
        ClassifierPrediction(model_name="TITAN", label="LUAD", confidence=0.8),
        ClassifierPrediction(model_name="MIZero", label="LUAD", confidence=0.7),
    ]
    import asyncio
    cv = asyncio.run(cv_agent._calculate_verification_scores(
        diagnosis={"diagnosis": "LUAD"},
        classifier_predictions=preds,
    ))
    assert cv.classifier_internal_consistency == 1.0
    # MCS = avg(1*0.9, 1*0.8, 1*0.7) = 0.8
    assert cv.multi_modal_consistency == pytest.approx(0.8)
    assert cv.classifier_verification_score == pytest.approx(0.8)


def test_classifier_internal_consistency_split(cv_agent, monkeypatch):
    async def fake_compare(self, a, b):
        return 1.0 if a.lower() == b.lower() else 0.0
    monkeypatch.setattr(ClassifierVerificationAgent, "compare_diagnoses", fake_compare)

    preds = [
        ClassifierPrediction(model_name="Conch", label="LUAD", confidence=0.9),
        ClassifierPrediction(model_name="TITAN", label="LUSC", confidence=0.8),
    ]
    import asyncio
    cv = asyncio.run(cv_agent._calculate_verification_scores(
        diagnosis={"diagnosis": "LUAD"},
        classifier_predictions=preds,
    ))
    # 1 of 2 classifiers agrees on majority label → CIS = 0.5
    assert cv.classifier_internal_consistency == 0.5
    # MCS = avg(1*0.9, 0*0.8) = 0.45
    assert cv.multi_modal_consistency == pytest.approx(0.45)
    # CVS = max(MCS * CIS, 0.01) = max(0.225, 0.01) = 0.225
    assert cv.classifier_verification_score == pytest.approx(0.225)


def test_classifier_cvs_floors_at_001(cv_agent, monkeypatch):
    async def fake_compare(self, a, b):
        return 0.0  # nothing matches
    monkeypatch.setattr(ClassifierVerificationAgent, "compare_diagnoses", fake_compare)

    preds = [
        ClassifierPrediction(model_name="Conch", label="LUAD", confidence=0.9),
    ]
    import asyncio
    cv = asyncio.run(cv_agent._calculate_verification_scores(
        diagnosis={"diagnosis": "LUSC"},
        classifier_predictions=preds,
    ))
    # MCS=0, CIS=1 → CVS = max(0, 0.01) = 0.01 (floor)
    assert cv.classifier_verification_score == pytest.approx(0.01)
