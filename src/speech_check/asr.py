"""Verbatim speech recognition with word timings and per-word confidence (CrisperWhisper 2.0).

CrisperWhisper keeps fillers, repetitions and cut-off words, which ordinary Whisper models drop. Its model
weights are under a non-commercial research licence (docs/licences.md). This module is the only place that
knows the ASR engine; another engine only has to return the same Word list.
"""
from __future__ import annotations

import numpy as np

from . import config
from .audio import SAMPLE_RATE
from .words import Word, make_word


class AsrError(RuntimeError):
    pass


def transcribe(samples: np.ndarray, lang: str, *, model: str = config.DEFAULT_ASR_MODEL) -> tuple[list[Word], dict]:
    from crisperwhisper import CrisperWhisperModel

    if model not in config.ASR_MODELS:
        raise AsrError(f"unknown CrisperWhisper size {model!r}; expected one of {sorted(config.ASR_MODELS)}")
    try:
        engine = CrisperWhisperModel(model, backend="ct2", device="cpu", compute_type="int8")
    except Exception as exc:  # the library raises plain exceptions for a missing or unconverted model
        raise AsrError(f"could not load CrisperWhisper {model}; run `speech-check setup` first ({exc})") from exc
    result = engine.transcribe(samples, sr=SAMPLE_RATE, language=lang, mode="verbatim", word_confidence=True)
    words = [make_word(i, w.word, w.start, w.end, getattr(w, "probability", None)) for i, w in enumerate(result.words or [])]
    return words, {"engine": "crisperwhisper", "model": config.ASR_MODELS[model], "compute_type": "int8"}
