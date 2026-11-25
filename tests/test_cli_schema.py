import json
import os

import jsonschema
import pytest

from speech_check import cli, config

SCHEMA = json.loads(cli.resources.files("speech_check").joinpath("result.schema.json").read_text(encoding="utf-8"))


def validate(result):
    jsonschema.Draft202012Validator(SCHEMA).validate(result)


def test_schema_subcommand_prints_valid_schema(capsys):
    assert cli.main(["schema"]) == 0
    jsonschema.Draft202012Validator.check_schema(json.loads(capsys.readouterr().out))


def test_analyze_forces_hub_offline(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    config.configure_environment(offline=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    config.configure_environment(offline=False)
    assert "HF_HUB_OFFLINE" not in os.environ


def test_missing_file_is_a_clean_error(capsys, tmp_path):
    assert cli.main(["analyze", str(tmp_path / "nope.ogg")]) == 1
    assert "ffmpeg could not read" in capsys.readouterr().err


def test_minimal_result_validates():
    validate({
        "schema_version": 1, "tool": {"name": "speech-check", "version": "0"},
        "input": {"file": "a.ogg", "duration_s": 1.0, "language": "en", "mode": "free_speech"},
        "recording": {"snr_db_estimate": 20.0, "warnings": []}, "models": {},
        "reliability": {k: {"level": "low", "note": "x"} for k in ("transcript", "fluency", "pronunciation", "intonation")},
        "transcript": {"text": "", "words": [], "sentences": []}, "fluency": {"available": False},
        "timing_s": {"total": 0.1}, "summary_text": "",
    })
    with pytest.raises(jsonschema.ValidationError):
        validate({"schema_version": 2})
