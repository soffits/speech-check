"""Fluency measures from word timings: rate, pauses, fillers, repetitions and self-corrections.

Definitions follow the speed and breakdown measures of L2 fluency research (speech rate, articulation
rate, mean length of run, silent pause frequency and length; silent pause threshold 0.25 s).
"""
from __future__ import annotations

from .phonetics import phones_for_words, syllable_count
from .words import Word, sentences

PAUSE_MIN = 0.25  # s: gaps at least this long count as silent pauses
LONG_PAUSE = 0.6  # s
# Words that cannot end a clause: a pause after one is inside a phrase even if the ASR wrote a full stop.
_DANGLING = {
    "en": {"a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "from", "by", "and", "but", "or", "so",
           "because", "that", "which", "who", "if", "when", "as", "my", "your", "his", "her", "our", "their", "is",
           "are", "was", "were", "be", "have", "has", "had", "will", "would", "can", "could", "i", "we", "they",
           "he", "she", "it", "than", "about", "into", "like"},
    "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "à", "au", "aux", "en", "et", "ou", "mais", "que",
           "qui", "dans", "pour", "avec", "sur", "par", "ma", "mon", "mes", "ta", "ton", "tes", "sa", "son", "ses",
           "notre", "votre", "leur", "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "ne", "est",
           "sont", "a", "ont", "si", "quand", "comme", "parce", "puis"},
}
_REPAIR_MARKERS = {
    "en": [("i", "mean"), ("sorry",), ("or", "rather"), ("no", "wait"), ("well", "actually"), ("let", "me", "rephrase")],
    "fr": [("enfin",), ("pardon",), ("je", "veux", "dire"), ("ou", "plutôt"), ("non", "attends")],
}


def analyze(words: list[Word], duration: float, lang: str) -> dict:
    spoken = [w for w in words if w.kind != "sound"]
    content = [w for w in spoken if w.kind in ("word", "cutoff")]
    if not content:
        return {"available": False, "reason": "no words were recognised"}

    start, end = spoken[0].start, spoken[-1].end
    span = max(end - start, 1e-6)
    raw = phones_for_words([w.norm or "x" for w in content], lang)
    syll = sum(syllable_count(p, w.norm) for p, w in zip(raw, content))

    pauses = _pauses(spoken, lang)
    silent = sum(p["duration"] for p in pauses)
    articulation_time = max(span - silent, 1e-6)
    runs = _runs(spoken, content, raw)

    long = [p for p in pauses if p["duration"] >= LONG_PAUSE]
    fillers = [w for w in spoken if w.kind == "filler"]
    n_words = sum(1 for w in content if w.kind == "word")
    minutes = span / 60

    return {
        "available": True,
        "duration_s": round(duration, 2),
        "speaking_span_s": round(span, 2),
        "leading_silence_s": round(start, 2),
        "trailing_silence_s": round(max(duration - end, 0.0), 2),
        "words": n_words,
        "syllables": syll,
        "speech_rate_wpm": round(n_words / minutes, 1),
        "speech_rate_syll_per_s": round(syll / span, 2),
        "articulation_rate_syll_per_s": round(syll / articulation_time, 2),
        "phonation_time_ratio": round(articulation_time / span, 3),
        "mean_length_of_run_syllables": round(sum(r["syllables"] for r in runs) / len(runs), 1) if runs else None,
        "mean_length_of_run_words": round(sum(r["words"] for r in runs) / len(runs), 1) if runs else None,
        "pauses": {
            "threshold_s": PAUSE_MIN,
            "long_threshold_s": LONG_PAUSE,
            "count": len(pauses),
            "per_minute": round(len(pauses) / minutes, 1),
            "mean_duration_s": round(silent / len(pauses), 2) if pauses else 0.0,
            "longest_s": round(max((p["duration"] for p in pauses), default=0.0), 2),
            "long_count": len(long),
            "long_by_position": _count_by(long, "position"),
            "by_position": _count_by(pauses, "position"),
            "long": long,
        },
        "fillers": {
            "count": len(fillers),
            "per_minute": round(len(fillers) / minutes, 1),
            "per_100_words": round(100 * len(fillers) / max(n_words, 1), 1),
            "items": [{"text": w.text, "start": round(w.start, 2)} for w in fillers],
        },
        "repetitions": _repetitions(spoken),
        "self_corrections": _self_corrections(spoken, lang),
        "sentences": len(sentences(words)),
    }


def _position(prev: Word, nxt: Word, lang: str) -> str:
    if prev.kind == "filler":
        return "after_filler"
    if prev.norm in _DANGLING.get(lang, ()):
        return "within_clause"
    return {"sentence": "between_sentences", "clause": "between_clauses"}.get(prev.boundary or "", "within_clause")


def _pauses(spoken: list[Word], lang: str) -> list[dict]:
    out = []
    for i in range(len(spoken) - 1):
        a, b = spoken[i], spoken[i + 1]
        gap = b.start - a.end
        if gap >= PAUSE_MIN:
            out.append({
                "start": round(a.end, 2), "end": round(b.start, 2), "duration": round(gap, 2),
                "position": _position(a, b, lang),
                "after_word": a.text, "before_word": b.text,
                "context": " ".join(w.text for w in spoken[max(0, i - 2): i + 1]) + " | " + " ".join(w.text for w in spoken[i + 1: i + 4]),
            })
    return out


def _count_by(pauses: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for p in pauses:
        counts[p[key]] = counts.get(p[key], 0) + 1
    return counts


def _runs(spoken: list[Word], content: list[Word], raw: list[list[str]]) -> list[dict]:
    """Stretches of speech between silent pauses; fillers inside a run belong to it but add no syllables."""
    syll_of = {id(w): syllable_count(p, w.norm) for w, p in zip(content, raw)}
    runs, cur = [], {"words": 0, "syllables": 0}
    for i, w in enumerate(spoken):
        if w.kind in ("word", "cutoff"):
            cur["words"] += 1
            cur["syllables"] += syll_of[id(w)]
        if i + 1 < len(spoken) and spoken[i + 1].start - w.end >= PAUSE_MIN:
            if cur["words"]:
                runs.append(cur)
            cur = {"words": 0, "syllables": 0}
    if cur["words"]:
        runs.append(cur)
    return runs


def _repetitions(spoken: list[Word]) -> dict:
    """Immediately repeated words or phrases (fillers in between are skipped): "I I", "to like to like"."""
    toks = [w for w in spoken if w.kind in ("word", "cutoff")]
    items, i = [], 0
    while i < len(toks):
        hit = None
        for n in (3, 2, 1):
            if i + 2 * n <= len(toks):
                a = [t.norm for t in toks[i: i + n]]
                b = [t.norm for t in toks[i + n: i + 2 * n]]
                if a == b and all(a) and toks[i].kind == "word":
                    hit = n
                    break
        if hit:
            n = hit
            unit = [t.norm for t in toks[i: i + n]]
            times = 2
            while i + (times + 1) * n <= len(toks) and [t.norm for t in toks[i + times * n: i + (times + 1) * n]] == unit:
                times += 1
            items.append({"text": " ".join(unit), "times": times, "start": round(toks[i].start, 2), "units": n})
            i += times * n
        else:
            i += 1
    return {"count": len(items), "items": items}


def _self_corrections(spoken: list[Word], lang: str) -> dict:
    """Cut-off words that are restarted ("th- Thursday") and explicit repair phrases ("I mean")."""
    items = []
    for i, w in enumerate(spoken):
        if w.kind == "cutoff":
            frag = w.norm.rstrip("-")
            nxt = next((x for x in spoken[i + 1: i + 4] if x.kind == "word"), None)
            restart = bool(nxt and frag and nxt.norm.startswith(frag[: max(1, len(frag) - 1)]))
            items.append({"type": "restart" if restart else "cut_off", "text": w.text,
                          "then": nxt.text if nxt else None, "start": round(w.start, 2)})
    norms = [w.norm for w in spoken]
    for marker in _REPAIR_MARKERS.get(lang, []):
        n = len(marker)
        for i in range(len(norms) - n + 1):
            if tuple(norms[i: i + n]) == marker:
                items.append({"type": "repair_phrase", "text": " ".join(marker), "start": round(spoken[i].start, 2)})
    items.sort(key=lambda x: x["start"])
    return {"count": len(items), "items": items}
