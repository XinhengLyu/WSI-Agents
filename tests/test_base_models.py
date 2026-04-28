"""Tests for pydantic data models."""
import pytest
from pydantic import ValidationError

from base_models import (
    AnalysisTask, ClassifierPrediction, ClassifierVerification,
    ClassifierVerificationResult, ConsistencyResult, IntegrationResult,
    KnowledgeValidation, ModelAnalysis, ModelReliability,
    TaskAllocation, TaskAllocationRequest, VerificationResult,
)


def test_model_reliability_accepts_none_classifier_score():
    """Treatment task: classifier not used."""
    r = ModelReliability(
        knowledge_score=0.8,
        consistency_score=0.7,
        classifier_score=None,
        final_score=0.76,
        assessment_details="x",
    )
    assert r.classifier_score is None


def test_model_reliability_accepts_float_classifier_score():
    r = ModelReliability(
        knowledge_score=0.8,
        consistency_score=0.7,
        classifier_score=0.6,
        final_score=0.7,
        assessment_details="x",
    )
    assert r.classifier_score == 0.6


def test_analysis_task_requires_all_three_responses():
    with pytest.raises(ValidationError):
        AnalysisTask(
            question_id="q1",
            question="?",
            mllm1_response="a",
            mllm2_response="b",
            # mllm3_response missing
        )


def test_classifier_prediction_round_trip():
    p = ClassifierPrediction(model_name="Conch", label="LUAD", confidence=0.92)
    assert p.confidence == 0.92


def test_consistency_result_with_three_models():
    a = ModelAnalysis(content_score=0.9, reasoning_score=0.8,
                     consistency_score=0.85, issues_found=["x"])
    cr = ConsistencyResult(
        content="{}",
        mllm1_analysis=a,
        mllm2_analysis=a,
        mllm3_analysis=a,
    )
    assert cr.mllm2_analysis.consistency_score == 0.85


def test_verification_result_with_three_models():
    v = KnowledgeValidation(score=0.7, matched_rules=["a"], unmatched_rules=[],
                            validation_details="ok")
    vr = VerificationResult(
        content="{}",
        mllm1_validation=v,
        mllm2_validation=v,
        mllm3_validation=v,
    )
    assert vr.mllm1_validation.score == 0.7


def test_classifier_verification_with_empty_predictions():
    cv = ClassifierVerification(
        multi_modal_consistency=0.0,
        classifier_internal_consistency=0.0,
        classifier_verification_score=0.0,
        classifier_predictions=[],
    )
    assert cv.classifier_predictions == []


def test_task_allocation_round_trip():
    req = TaskAllocationRequest(question_id="q1", question="?")
    assert req.question_id == "q1"
    alloc = TaskAllocation(agent_type="diagnosis", question_id="q1", question="?")
    assert alloc.agent_type == "diagnosis"


def test_integration_result_holds_all_three_reliabilities():
    r = ModelReliability(knowledge_score=0.5, consistency_score=0.5,
                         classifier_score=None, final_score=0.5,
                         assessment_details="x")
    ir = IntegrationResult(
        content="{}",
        mllm1_reliability=r, mllm2_reliability=r, mllm3_reliability=r,
        final_conclusion="text", supporting_evidence=[],
    )
    assert ir.mllm3_reliability.final_score == 0.5
