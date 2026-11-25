"""The whole pipeline: audio file in, result dict out."""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import SCHEMA_VERSION, __version__, asr, audio, config, fluency, intonation, pronunciation, reliability, report
from .words import sentence_text, sentences


def run(path: str, *, lang: str = "en", reference: str | None = None,
        asr_model: str = config.DEFAULT_ASR_MODEL, pronunciation_on: bool = True, intonation_on: bool = True,
        threads: int | None = None, log=lambda msg: None) -> dict:
    if lang not in config.LANGUAGES:
        raise ValueError(f"unsupported language {lang!r}; expected one of {config.LANGUAGES}")
    timing: dict[str, float] = {}
    t_start = time.time()

    def stage(name: str, fn):
        log(f"{name}...")
        t = time.time()
        value = fn()
        timing[name] = round(time.time() - t, 1)
        return value

    clip = stage("decode", lambda: audio.load(path))
    quality = audio.quality(clip.samples)
    words, asr_info = stage("asr", lambda: asr.transcribe(clip.samples, lang, model=asr_model))
    if not words:
        raise RuntimeError("no speech was recognised in the recording")
    mode = "read_aloud" if reference else "free_speech"

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "speech-check", "version": __version__},
        "input": {"file": Path(path).name, "duration_s": round(clip.duration, 2), "language": lang, "mode": mode,
                  "reference_text": reference, "source": clip.source},
        "recording": {**quality, "warnings": audio.quality_warnings(clip.duration, quality)},
        "models": {"asr": asr_info, "phones": config.PHONE_MODEL if pronunciation_on else None,
                   "pitch": "parselmouth (Praat)" if intonation_on else None},
        "reliability": reliability.notes(lang, mode, quality["snr_db_estimate"]),
        "transcript": {
            "text": " ".join(w.text for w in words),
            "words": [w.to_dict() for w in words],
            "sentences": [{"text": sentence_text(s), "start": round(s[0].start, 2), "end": round(s[-1].end, 2)} for s in sentences(words)],
        },
    }
    result["fluency"] = stage("fluency", lambda: fluency.analyze(words, clip.duration, lang))
    if pronunciation_on:
        def pron():
            from .phone_model import PhoneModel
            return pronunciation.analyze(clip.samples, words, lang, PhoneModel(threads), reference)
        result["pronunciation"] = stage("pronunciation", pron)
    if intonation_on:
        result["intonation"] = stage("intonation", lambda: intonation.analyze(clip.samples, words))
    timing["total"] = round(time.time() - t_start, 1)
    result["timing_s"] = timing
    result["summary_text"] = report.summary(result)
    return result
