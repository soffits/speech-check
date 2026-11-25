"""The word record shared by every analysis step, and how raw ASR tokens are classified."""
from __future__ import annotations

import re
from dataclasses import dataclass

FILLERS = {"um", "uh", "er", "erm", "hm", "hmm", "mm", "mhm", "uhm", "euh", "heu", "hum"}
_BRACKETED = re.compile(r"^\[(.+)\]$")
_SENTENCE_END = ".?!…"
_CLAUSE_END = ",;:"


@dataclass
class Word:
    index: int
    text: str  # as the ASR wrote it, punctuation included
    start: float
    end: float
    confidence: float | None  # ASR probability, 0-1
    kind: str  # word | filler | cutoff | sound

    @property
    def norm(self) -> str:
        """Lower-case letters, apostrophes and inner hyphens only."""
        return normalize(self.text)

    @property
    def boundary(self) -> str | None:
        """'sentence' or 'clause' when the ASR put punctuation after the word."""
        stripped = self.text.rstrip("\"')]”’ ")
        if stripped and stripped[-1] in _SENTENCE_END:
            return "sentence"
        if stripped and stripped[-1] in _CLAUSE_END:
            return "clause"
        return None

    def to_dict(self) -> dict:
        return {"i": self.index, "text": self.text, "start": round(self.start, 3), "end": round(self.end, 3),
                "confidence": None if self.confidence is None else round(self.confidence, 3), "kind": self.kind}


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w'\-]", "", text, flags=re.UNICODE).replace("_", "")
    return text.strip("-'")


def classify(raw: str) -> str:
    """word | filler | cutoff | sound, from one raw token of a verbatim transcript."""
    token = raw.strip()
    bracket = _BRACKETED.match(token.rstrip(".,?!"))
    if bracket:
        return "filler" if normalize(bracket.group(1)) in FILLERS else "sound"
    bare = token.rstrip(".,;:?!…\"'”’ ")
    if len(bare) > 1 and bare.endswith(("-", "–")):
        return "cutoff"
    if normalize(token) in FILLERS:
        return "filler"
    return "word"


def make_word(index: int, raw: str, start: float, end: float, confidence: float | None) -> Word:
    kind = classify(raw)
    text = raw.strip()
    return Word(index, text, float(start), max(float(end), float(start)), confidence, kind)


def sentences(words: list[Word]) -> list[list[Word]]:
    """Split spoken tokens (sounds excluded) into sentences at the ASR's sentence punctuation."""
    out: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if w.kind == "sound":
            continue
        current.append(w)
        if w.boundary == "sentence" and w.kind != "filler":
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def sentence_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)
