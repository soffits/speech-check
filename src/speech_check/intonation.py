"""Pitch contour statistics per sentence, in semitones, from Praat's autocorrelation tracker (parselmouth).

Pitch is tracked in two passes: a wide first pass estimates the speaker's range, a second pass uses limits
derived from it (the standard way to avoid octave errors without knowing the speaker's sex). Remaining
octave jumps are folded back, isolated frames dropped, and the contour lightly smoothed.
"""
from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE
from .words import Word, sentence_text, sentences

STEP = 0.01  # s
MIN_VOICED_S = 0.6  # a sentence needs this much voiced speech for stable statistics
FLAT_RANGE_ST = 5.0
FLAT_SD_ST = 1.5
FINAL_MOVE_ST = 2.0


def analyze(samples: np.ndarray, words: list[Word]) -> dict:
    contour = _contour(samples)
    if contour is None:
        return {"available": False, "reason": "no voiced speech found"}
    times, st, hz = contour
    results = []
    for sent in sentences(words):
        speech = [w for w in sent if w.kind in ("word", "cutoff")]
        if not speech:
            continue
        results.append(_sentence(sent, speech, times, st, hz))

    voiced = ~np.isnan(st)
    whole = _stats(st[voiced]) if voiced.sum() * STEP >= MIN_VOICED_S else None
    usable = [r for r in results if r.get("reliable")]
    return {
        "available": True,
        "method": "parselmouth (Praat) autocorrelation, two-pass limits, octave-jump cleaning",
        "speaker_median_hz": round(float(np.nanmedian(hz)), 1),
        "overall": whole,
        "sentences": results,
        "summary": {
            "sentences_total": len(results),
            "sentences_with_stable_pitch": len(usable),
            "median_sentence_range_st": _median([r["range_st"] for r in usable]),
            "median_sentence_sd_st": _median([r["sd_st"] for r in usable]),
            "flat_sentences": sum(1 for r in usable if r["flat"]),
        },
    }


def _median(values: list[float]):
    return round(float(np.median(values)), 2) if values else None


def _contour(samples: np.ndarray):
    import parselmouth

    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    first = snd.to_pitch_ac(time_step=STEP, pitch_floor=60, pitch_ceiling=600, very_accurate=False)
    f0 = first.selected_array["frequency"]
    voiced = f0[f0 > 0]
    if len(voiced) < 30:
        return None
    q1, q3 = np.percentile(voiced, [25, 75])
    floor, ceiling = max(50.0, 0.75 * q1), min(700.0, 2.5 * q3)
    pitch = snd.to_pitch_ac(time_step=STEP, pitch_floor=floor, pitch_ceiling=ceiling, very_accurate=True,
                            octave_jump_cost=0.6, voicing_threshold=0.5)
    times = pitch.xs()
    hz = pitch.selected_array["frequency"].astype(float)
    hz[hz <= 0] = np.nan
    if np.isnan(hz).all():
        return None
    st = _clean(12 * np.log2(hz / np.nanmedian(hz)))
    return times, st, np.where(np.isnan(st), np.nan, np.nanmedian(hz) * 2 ** (st / 12))


def _clean(st: np.ndarray) -> np.ndarray:
    """Fold octave errors toward the local median, drop isolated frames, smooth with a 5-frame median."""
    st = st.copy()
    n = len(st)
    half = 25
    ref = np.full(n, np.nan)
    for i in range(n):
        window = st[max(0, i - half): i + half + 1]
        if np.sum(~np.isnan(window)) >= 5:
            ref[i] = np.nanmedian(window)
    for i in range(n):
        if np.isnan(st[i]) or np.isnan(ref[i]):
            continue
        for shift in (-12.0, 12.0, -24.0, 24.0):
            if abs(st[i] - ref[i]) > 7 and abs(st[i] + shift - ref[i]) < 4:
                st[i] += shift
                break
        if abs(st[i] - ref[i]) > 9:
            st[i] = np.nan
    # drop voiced runs shorter than 5 frames
    i = 0
    while i < n:
        if np.isnan(st[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(st[j]):
            j += 1
        if j - i < 5:
            st[i:j] = np.nan
        else:
            seg = st[i:j]
            padded = np.pad(seg, 2, mode="edge")
            st[i:j] = np.array([np.median(padded[k: k + 5]) for k in range(len(seg))])
        i = j
    return st


def _stats(st: np.ndarray) -> dict:
    p5, p95 = np.percentile(st, [5, 95])
    return {"range_st": round(float(p95 - p5), 2), "sd_st": round(float(np.std(st)), 2), "voiced_s": round(len(st) * STEP, 2)}


def _sentence(sent: list[Word], speech: list[Word], times, st, hz) -> dict:
    a, b = sent[0].start, sent[-1].end
    mask = (times >= a) & (times <= b) & ~np.isnan(st)
    seg = st[mask]
    out = {"text": sentence_text(sent), "start": round(a, 2), "end": round(b, 2),
           "ends_with": "question" if sent[-1].text.rstrip().endswith("?") else "statement"}
    voiced_s = len(seg) * STEP
    out["voiced_s"] = round(voiced_s, 2)
    if voiced_s < MIN_VOICED_S:
        out["reliable"] = False
        out["note"] = "too little voiced speech for stable pitch statistics"
        return out
    s = _stats(seg)
    out.update(s)
    out["median_hz"] = round(float(np.nanmedian(hz[mask])), 1)
    out["flat"] = s["range_st"] < FLAT_RANGE_ST and s["sd_st"] < FLAT_SD_ST
    move = _final_movement(times, st, speech[-1])
    out["final_movement_st"] = move
    out["final_contour"] = None if move is None else ("rising" if move >= FINAL_MOVE_ST else "falling" if move <= -FINAL_MOVE_ST else "level")
    out["reliable"] = True
    return out


def _final_movement(times, st, last: Word):
    """Pitch at the end of the sentence's last word minus the pitch earlier in that word and the one before it."""
    a = last.start - 0.4
    end_mask = (times >= max(last.end - 0.15, last.start)) & (times <= last.end) & ~np.isnan(st)
    before_mask = (times >= a) & (times < max(last.end - 0.15, last.start)) & ~np.isnan(st)
    if end_mask.sum() < 3 or before_mask.sum() < 5:
        return None
    return round(float(np.median(st[end_mask]) - np.median(st[before_mask])), 2)
