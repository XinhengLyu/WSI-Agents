"""Tests for ScoreCalculator."""
import pytest

from ScoreCalculator import ScoreCalculator


def test_final_score_with_classifier_uses_3_2_5_weighting():
    score = ScoreCalculator.calculate_final_score(
        knowledge_score=1.0,
        consistency_score=1.0,
        classifier_score=1.0,
    )
    assert score == pytest.approx(1.0)

    # Knowledge=1, others=0 → 0.3
    score = ScoreCalculator.calculate_final_score(1.0, 0.0, 0.0)
    assert score == pytest.approx(0.3)

    # Consistency=1, others=0 → 0.2
    score = ScoreCalculator.calculate_final_score(0.0, 1.0, 0.0)
    assert score == pytest.approx(0.2)

    # Classifier=1, others=0 → 0.5
    score = ScoreCalculator.calculate_final_score(0.0, 0.0, 1.0)
    assert score == pytest.approx(0.5)


def test_final_score_without_classifier_uses_6_4_reweight():
    """When classifier_score is None (Treatment task or missing data)."""
    score = ScoreCalculator.calculate_final_score(
        knowledge_score=1.0,
        consistency_score=1.0,
        classifier_score=None,
    )
    assert score == pytest.approx(1.0)

    # Knowledge=1, consistency=0 → 0.6
    score = ScoreCalculator.calculate_final_score(1.0, 0.0, None)
    assert score == pytest.approx(0.6)

    # Knowledge=0, consistency=1 → 0.4
    score = ScoreCalculator.calculate_final_score(0.0, 1.0, None)
    assert score == pytest.approx(0.4)


def test_final_score_clamps_to_unit_interval():
    # Even with extreme inputs, output stays in [0, 1]
    assert ScoreCalculator.calculate_final_score(2.0, 2.0, 2.0) == 1.0
    assert ScoreCalculator.calculate_final_score(-1.0, -1.0, -1.0) == 0.0


def test_semantic_score_handles_zero_relations():
    assert ScoreCalculator.calculate_semantic_score(0, 0) == 0.0


def test_semantic_score_basic():
    assert ScoreCalculator.calculate_semantic_score(10, 2) == 0.8
    assert ScoreCalculator.calculate_semantic_score(10, 0) == 1.0


def test_logical_score_basic():
    assert ScoreCalculator.calculate_logical_score(10, 5) == 0.5


def test_consistency_score_average():
    assert ScoreCalculator.calculate_consistency_score(0.4, 0.8) == pytest.approx(0.6)


def test_knowledge_score_basic():
    assert ScoreCalculator.calculate_knowledge_score(3, 4) == 0.75
    assert ScoreCalculator.calculate_knowledge_score(0, 4) == 0.0
    assert ScoreCalculator.calculate_knowledge_score(0, 0) == 0.0
