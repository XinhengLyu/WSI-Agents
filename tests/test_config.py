"""Tests for Config class."""
import json
import os

import pytest

from config import Config


@pytest.fixture
def tmp_output(tmp_path, monkeypatch):
    """Redirect Config paths into a tmp dir for the duration of the test."""
    monkeypatch.setattr(Config, "OUTPUT_DIR",      str(tmp_path / "output"))
    monkeypatch.setattr(Config, "QUESTIONS_DIR",   str(tmp_path / "questions"))
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.QUESTIONS_DIR, exist_ok=True)
    yield tmp_path


def test_configure_task_sets_state(tmp_output):
    Config.configure_task("Diagnosis", "Diagnosis-questions.jsonl")
    assert Config.TASK_NAME == "Diagnosis"
    assert Config.QUESTIONS_PATH.endswith("Diagnosis-questions.jsonl")
    assert Config.REFINED_RESPONSES_PATH.endswith("refined_responses_Diagnosis.jsonl")


def test_configure_task_rejects_invalid_name(tmp_output):
    with pytest.raises(ValueError, match="Unknown task"):
        Config.configure_task("Bogus", "x.jsonl")


def test_get_processed_ids_returns_empty_when_missing(tmp_output, monkeypatch):
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH",
                        str(tmp_output / "does_not_exist.jsonl"))
    assert Config.get_processed_ids() == set()


def test_get_processed_ids_reads_jsonl(tmp_output, monkeypatch):
    path = tmp_output / "out.jsonl"
    path.write_text(
        json.dumps({"question_id": "a", "text": "x"}) + "\n" +
        json.dumps({"question_id": "b", "text": "y"}) + "\n"
    )
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH", str(path))
    assert Config.get_processed_ids() == {"a", "b"}


def test_get_processed_ids_warns_on_malformed(tmp_output, monkeypatch, capsys):
    path = tmp_output / "out.jsonl"
    path.write_text(
        json.dumps({"question_id": "a"}) + "\n" +
        "{not valid json\n"
    )
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH", str(path))
    result = Config.get_processed_ids()
    assert result == {"a"}
    captured = capsys.readouterr()
    assert "Warning" in captured.out


def test_update_refined_responses_path_with_bare_filename(tmp_output, monkeypatch):
    """Should not raise when path has no parent directory."""
    monkeypatch.chdir(tmp_output)
    Config.update_refined_responses_path("bare.jsonl")
    assert Config.REFINED_RESPONSES_PATH == "bare.jsonl"


def test_update_refined_responses_path_creates_parent(tmp_output):
    nested = tmp_output / "deep" / "nested" / "out.jsonl"
    Config.update_refined_responses_path(str(nested))
    assert os.path.isdir(os.path.dirname(str(nested)))
