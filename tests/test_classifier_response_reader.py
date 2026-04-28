"""Tests for ClassifierResponseReader."""
import json
import pytest

from ClassifierResponseReader import ClassifierResponseReader


@pytest.fixture
def reader(tmp_path):
    """Two classifier files keyed by slide id."""
    conch = tmp_path / "Conch.jsonl"
    titan = tmp_path / "TITAN.jsonl"
    conch.write_text(
        json.dumps({"question_id": "TCGA-XYZ", "label": "LUAD",  "confidence": 0.92}) + "\n"
    )
    titan.write_text(
        json.dumps({"question_id": "TCGA-XYZ", "label": "LUSC",  "confidence": 0.71}) + "\n"
    )
    return ClassifierResponseReader(base_path=str(tmp_path) + "/")


def test_extract_search_key_strips_prefix_and_extension(reader):
    """question_id "Diagnosis_TCGA-XYZ.svs" → key "TCGA-XYZ"."""
    preds = reader.get_predictions(["Conch", "TITAN"], "Diagnosis_TCGA-XYZ.svs")
    assert len(preds) == 2
    labels = sorted(p.label for p in preds)
    assert labels == ["LUAD", "LUSC"]


def test_returns_empty_when_no_match(reader):
    preds = reader.get_predictions(["Conch", "TITAN"], "Diagnosis_UNKNOWN.svs")
    assert preds == []


def test_predictions_carry_model_name(reader):
    preds = reader.get_predictions(["Conch"], "Diagnosis_TCGA-XYZ.svs")
    assert len(preds) == 1
    assert preds[0].model_name == "Conch"
    assert preds[0].confidence == pytest.approx(0.92)


def test_loads_each_model_file_only_once(reader):
    reader.get_predictions(["Conch"], "Diagnosis_TCGA-XYZ.svs")
    reader.get_predictions(["Conch"], "Diagnosis_TCGA-XYZ.svs")
    assert "Conch" in reader.predictions


def test_missing_classifier_file_is_skipped(tmp_path, capsys):
    r = ClassifierResponseReader(base_path=str(tmp_path) + "/")
    preds = r.get_predictions(["DoesNotExist"], "Diagnosis_X.svs")
    assert preds == []
