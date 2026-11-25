"""Real models end to end. Skipped unless the models are set up and a Piper voice is available.

Run with: SPEECH_CHECK_HOME=<cache> SPEECH_CHECK_VOICES=<dir with en_US-amy-medium.onnx> pytest -m integration
"""
import os
from pathlib import Path

import pytest

from speech_check import config

pytestmark = pytest.mark.integration
VOICES = Path(os.environ.get("SPEECH_CHECK_VOICES", "/nonexistent"))


@pytest.mark.skipif(not (VOICES / "en_US-amy-medium.onnx").exists() or not config.home().joinpath("crisperwhisper").exists(),
                    reason="needs set-up models and a Piper voice")
def test_planted_clip_end_to_end(tmp_path):
    import jsonschema
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import make_clips
    from piper import PiperVoice
    from speech_check import analyze, cli

    config.configure_environment(offline=True)
    voice = PiperVoice.load(VOICES / "en_US-amy-medium.onnx")
    wav = tmp_path / "c.wav"
    import soundfile as sf
    sf.write(wav, make_clips.build(voice, [("Um", 0.4), ("we went for a long", 1.4), ("walk by the river.", 0.3),
                                           ("I think very few people came, and I think it was fine.", 0.2)]), 16000)
    result = analyze.run(str(wav), lang="en", intonation_on=True)
    schema = json.loads(cli.resources.files("speech_check").joinpath("result.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(result)
    assert result["fluency"]["fillers"]["count"] >= 1
    assert result["fluency"]["pauses"]["long_count"] >= 1
    assert result["pronunciation"]["available"]
