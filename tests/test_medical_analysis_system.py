"""Tests for MedicalAnalysisSystem helpers."""
import json
import pytest

from MedicalAnalysisSystem import read_jsonl
from config import Config


def test_read_jsonl_returns_all_cases(tmp_path, monkeypatch):
    p = tmp_path / "questions.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "prompt": "?"}) + "\n" +
        json.dumps({"question_id": "q2", "prompt": "?"}) + "\n"
    )
    # Force fresh state: no processed IDs
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH",
                        str(tmp_path / "no_output.jsonl"))
    cases = read_jsonl(str(p))
    assert len(cases) == 2
    assert {c["id"] for c in cases} == {"q1", "q2"}


def test_read_jsonl_skips_processed_ids(tmp_path, monkeypatch):
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        json.dumps({"question_id": "q1", "prompt": "?"}) + "\n" +
        json.dumps({"question_id": "q2", "prompt": "?"}) + "\n"
    )
    output = tmp_path / "out.jsonl"
    output.write_text(json.dumps({"question_id": "q1", "text": "done"}) + "\n")
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH", str(output))

    cases = read_jsonl(str(questions))
    assert [c["id"] for c in cases] == ["q2"]


def test_read_jsonl_skip_processed_disabled(tmp_path, monkeypatch):
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"question_id": "q1", "prompt": "?"}) + "\n")
    output = tmp_path / "out.jsonl"
    output.write_text(json.dumps({"question_id": "q1", "text": "done"}) + "\n")
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH", str(output))

    cases = read_jsonl(str(questions), skip_processed=False)
    assert [c["id"] for c in cases] == ["q1"]


def test_read_jsonl_handles_invalid_json_lines(tmp_path, monkeypatch, capsys):
    p = tmp_path / "questions.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "prompt": "?"}) + "\n" +
        "not valid json\n" +
        json.dumps({"question_id": "q2", "prompt": "?"}) + "\n"
    )
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH",
                        str(tmp_path / "no_output.jsonl"))
    cases = read_jsonl(str(p))
    assert {c["id"] for c in cases} == {"q1", "q2"}
    captured = capsys.readouterr()
    assert "Skipping invalid line" in captured.out


def test_read_jsonl_missing_file_returns_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Config, "REFINED_RESPONSES_PATH",
                        str(tmp_path / "no_output.jsonl"))
    cases = read_jsonl(str(tmp_path / "nope.jsonl"))
    assert cases == []
    captured = capsys.readouterr()
    assert "File not found" in captured.out
