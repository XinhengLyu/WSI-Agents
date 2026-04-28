"""Tests for JsonlResponseReader."""
import json
import pytest

from JsonlResponseReader import JsonlResponseReader


@pytest.fixture
def two_model_files(tmp_path):
    a = tmp_path / "model_a.jsonl"
    b = tmp_path / "model_b.jsonl"
    a.write_text(
        json.dumps({"question_id": "q1", "text": "answer-a-1"}) + "\n" +
        json.dumps({"question_id": "q2", "text": "answer-a-2"}) + "\n"
    )
    b.write_text(
        json.dumps({"question_id": "q1", "text": "answer-b-1"}) + "\n"
    )
    return {"A": str(a), "B": str(b)}


def test_loads_responses_from_each_file(two_model_files):
    reader = JsonlResponseReader(two_model_files)
    assert reader.get_response("A", "q1") == "answer-a-1"
    assert reader.get_response("A", "q2") == "answer-a-2"
    assert reader.get_response("B", "q1") == "answer-b-1"


def test_get_response_returns_none_for_unknown_question(two_model_files):
    reader = JsonlResponseReader(two_model_files)
    assert reader.get_response("A", "qXX") is None


def test_get_response_returns_none_for_unknown_model(two_model_files):
    reader = JsonlResponseReader(two_model_files)
    assert reader.get_response("Z", "q1") is None


def test_skips_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        "\n" + json.dumps({"question_id": "q1", "text": "ok"}) + "\n\n"
    )
    reader = JsonlResponseReader({"X": str(p)})
    assert reader.get_response("X", "q1") == "ok"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        JsonlResponseReader({"X": str(tmp_path / "nope.jsonl")})
