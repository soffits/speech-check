"""Phone-level pronunciation hints.

For each spoken word we know which phones espeak-ng expects (from the ASR transcript, or from a given
reference text in read-aloud mode) and which phones the wav2vec2 recogniser actually heard. The two are
aligned per audio segment; each expected phone also gets a CTC forced-alignment region in which its
posterior is measured (a goodness-of-pronunciation style plausibility). A wrong phone is only reported
with high confidence when the expected phone was also implausible where it should have been.

The confidence logic (near-phone costs, final-phone discounts, plausibility scaling, thresholds) is
adapted from OpenPronounce (MIT), which was benchmarked on read speech; see NOTICE.md. What is new here:
free speech (expected phones come from the transcript), fillers and cut-off words masked out of the audio
before decoding, forced-alignment regions for the plausibility, and repeated-pattern aggregation.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import Levenshtein

from .audio import SAMPLE_RATE
from .ctc import forced_align, greedy_decode
from .phone_model import FRAME_SECONDS, PhoneModel
from .phonetics import (ALTERNATES, normalize_phone, normalize_phones, phones_for_words, substitution_cost)
from .words import Word, normalize

REPORT_CONFIDENCE = 0.5  # a word is reported when its summed phone error confidences reach this
NEAR_COST = 0.5
FINAL_EXTRA_COST = 0.25
FINAL_DELETION_COST = 0.5
PLAUSIBLE_POSTERIOR = 0.2
LIKELY_WORD_CONFIDENCE = 0.9
PATTERN_PHONE_CONFIDENCE = 0.5
CONTEXT_FREE_MIN_PHONES = 3  # shorter words are too unreliable out of context
CONTEXT_FREE_WEIGHT = 0.8  # the context-free check is noisier than the in-context decode
CONTEXT_FREE_PAD_S = 0.03
CONTEXT_FREE_GAP_S = 0.3
SEGMENT_MAX_S = 22.0
SEGMENT_MIN_S = 10.0
SEGMENT_GAP_S = 0.3


@dataclass
class Target:
    word: str  # shown word
    expected: str  # normalised text whose phones are expected
    asr: Word
    replaced: bool = False  # read-aloud: the speaker said a different word (a text-level error, not a sound error)


def analyze(samples: np.ndarray, words: list[Word], lang: str, model: PhoneModel, reference: str | None = None) -> dict:
    targets, reference_report = _plan_targets(words, reference)
    if not targets:
        return {"available": False, "reason": "no words to assess"}
    target_by_asr = {t.asr.index: t for t in targets}
    groups = _token_groups(model.vocab, lang)

    reports, total_edits, total_expected = [], 0, 0
    for segment in _segments(words):
        r = _analyze_segment(samples, segment, target_by_asr, lang, model, groups)
        reports.extend(r["words"])
        total_edits += r["edits"]
        total_expected += r["expected"]

    _context_free_check(samples, reports, model, groups, lang)
    patterns = _patterns(reports)
    repeated = {(p["expected"], p["heard"]) for p in patterns if p["repeated"]}
    for rep in reports:
        _set_flag(rep, repeated, lang)
    flagged = [r for r in reports if r["flag"]]
    out = {
        "available": True,
        "mode": "read_aloud" if reference else "free_speech",
        "words": [_public(r) for r in reports],
        "flagged": [_public(r, with_phones=True) for r in flagged],
        "patterns": patterns,
        "overall": {
            "words_assessed": len(reports),
            "words_flagged_likely": sum(1 for r in flagged if r["flag"] == "likely"),
            "words_flagged_possible": sum(1 for r in flagged if r["flag"] == "possible"),
            "phone_error_rate": round(total_edits / total_expected, 3) if total_expected else None,
        },
    }
    if reference_report:
        out["reference"] = reference_report
    return out


# ---- targets and segments -------------------------------------------------------------------------

def _plan_targets(words: list[Word], reference: str | None) -> tuple[list[Target], dict | None]:
    spoken = [w for w in words if w.kind == "word" and w.norm]
    if not reference:
        return [Target(w.norm, w.norm, w) for w in spoken], None
    ref = [(tok, normalize(tok)) for tok in reference.split()]
    ref = [(_bare(tok), n) for tok, n in ref if n]
    sm = difflib.SequenceMatcher(None, [n for _, n in ref], [w.norm for w in spoken], autojunk=False)
    targets, omitted, inserted, substituted = [], [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                tok, norm = ref[i1 + k]
                targets.append(Target(tok, norm, spoken[j1 + k], replaced=tag == "replace"))
                if tag == "replace":
                    substituted.append({"expected": tok, "said": _bare(spoken[j1 + k].text)})
            omitted += [ref[i][0] for i in range(i1 + n, i2)]
            inserted += [_bare(spoken[j].text) for j in range(j1 + n, j2)]
        elif tag == "delete":
            omitted += [ref[i][0] for i in range(i1, i2)]
        else:
            inserted += [_bare(spoken[j].text) for j in range(j1, j2)]
    matched = len(ref) - len(omitted) - len(substituted)
    report = {
        "words_in_reference": len(ref),
        "matched": matched,
        "word_accuracy": round(matched / len(ref), 3) if ref else None,
        "substituted": substituted,
        "omitted": omitted,
        "inserted": inserted,
    }
    return targets, report


def _bare(text: str) -> str:
    return text.strip(".,;:?!\"“”()")


def _segments(words: list[Word]) -> list[list[Word]]:
    """Cut the token stream at pauses so each piece is at most ~22 s (wav2vec2 memory is quadratic in length)."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if i + 1 == len(words):
            break
        gap = words[i + 1].start - w.end
        length = w.end - cur[0].start
        if (length >= SEGMENT_MIN_S and gap >= SEGMENT_GAP_S) or length >= SEGMENT_MAX_S:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# ---- per-segment analysis -------------------------------------------------------------------------

def _token_groups(vocab: tuple[str, ...], lang: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, tok in enumerate(vocab):
        if tok.startswith("<") and tok.endswith(">"):
            continue
        groups.setdefault(normalize_phone(tok, lang), []).append(i)
    return groups


def _analyze_segment(samples, segment, target_by_asr, lang, model, groups) -> dict:
    t0 = max(segment[0].start - 0.2, 0.0)
    t1 = min(segment[-1].end + 0.2, len(samples) / SAMPLE_RATE)
    lp = model.log_probs(samples[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)]).copy()
    n_frames = lp.shape[0]
    for w in segment:  # fillers, cut-offs, laughs and unmatched words are not assessed: silence them
        if w.index not in target_by_asr:
            a = max(int((w.start - t0) / FRAME_SECONDS), 0)
            b = min(int(np.ceil((w.end - t0) / FRAME_SECONDS)), n_frames)
            lp[a:b, :] = -30.0
            lp[a:b, model.blank] = 0.0

    targets = [target_by_asr[w.index] for w in segment if w.index in target_by_asr]
    if not targets:
        return {"words": [], "edits": 0, "expected": 0}
    expected_by_word = [normalize_phones(p, lang) for p in phones_for_words([t.expected or "x" for t in targets], lang)]
    flat = [p for g in expected_by_word for p in g]
    heard = _heard_phones(lp, model, lang)

    emissions = _emissions(lp, model.blank, groups, flat)
    spans = forced_align(emissions, flat)
    regions = _regions(spans, n_frames) if spans else None
    posterior = [float(np.exp(emissions[a:b, k + 1].max())) if b > a else 0.0 for k, (a, b) in enumerate(regions)] if regions else None

    alignment = _align(flat, [h["phone"] for h in heard])
    words, offset = [], 0
    previous_last = None
    for target, group in zip(targets, expected_by_word):
        if not group:
            continue
        idx = list(range(offset, offset + len(group)))
        offset += len(group)
        matched = sorted(set().union(*(alignment[i] for i in idx)))
        actual = [heard[j]["phone"] for j in matched]
        candidates = _pronunciations(target.expected, group, previous_last, lang)
        distances = [Levenshtein.distance(c, actual) for c in candidates]
        distance = min(distances)
        candidate = candidates[distances.index(distance)]
        previous_last = group[-1]

        word_region = (regions[idx[0]][0], regions[idx[-1]][1]) if regions else (0, 0)
        per_phone_region = [regions[i] for i in idx] if (regions and candidate == group) else None
        reports = _phone_reports(candidate, actual, lp, groups, word_region, per_phone_region, lang)
        words.append({
            "target": target, "expected": group, "candidate": candidate, "heard": actual, "phones": reports,
            "edits": sum(p["error_confidence"] for p in reports), "context_free_heard": None,
        })
    edits_total = Levenshtein.distance(flat, [h["phone"] for h in heard])
    return {"words": words, "edits": edits_total, "expected": len(flat)}


def _heard_phones(lp, model, lang) -> list[dict]:
    tokens = greedy_decode(lp, model.vocab, model.blank)
    merged: list[dict] = []
    for tok in tokens:
        phone = normalize_phone(tok["token"], lang)
        if not phone:
            continue
        if merged and (merged[-1]["phone"] == phone or (merged[-1]["phone"] == "ɚ" and phone == "ɹ")):
            merged[-1]["confidence"] = max(merged[-1]["confidence"], tok["confidence"])
            continue
        merged.append({"phone": phone, "confidence": tok["confidence"]})
    return merged


def _emissions(lp, blank, groups, phones) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    cols = [lp[:, blank]]
    for p in phones:
        if p not in cache:
            ids = groups.get(p)
            cache[p] = lp[:, ids].max(axis=1) if ids else np.full(lp.shape[0], -20.0, dtype=lp.dtype)
        cols.append(cache[p])
    return np.stack(cols, axis=1)


def _regions(spans, n_frames) -> list[tuple[int, int]]:
    """Each phone's span widened to the midpoints between neighbours, so blanks around it count as its own."""
    regions = []
    for k, (a, b) in enumerate(spans):
        left = a if k == 0 else (spans[k - 1][1] + a) // 2
        right = b if k == len(spans) - 1 else (b + spans[k + 1][0]) // 2
        regions.append((left, max(right, left + 1)))
    return regions


def _align(expected: list[str], heard: list[str]) -> list[set[int]]:
    """Map every expected phone index to the heard phone indices it lines up with (insertions go to the previous phone)."""
    alignment: list[set[int]] = [set() for _ in expected]
    for op in Levenshtein.opcodes(expected, heard):
        tag, i1, i2, j1, j2 = op.tag, op.src_start, op.src_end, op.dest_start, op.dest_end
        if tag == "equal":
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment[k].add(l)
        elif tag == "replace":
            li, lj = i2 - i1, j2 - j1
            for k in range(i1, i2):
                s = j1 + int((k - i1) * lj / li)
                e = j1 + int((k - i1 + 1) * lj / li)
                alignment[k].update(range(s, e) if s != e else [min(s, j2 - 1)])
        elif tag == "insert":
            k = i1 - 1 if i1 > 0 else i1
            if k < len(alignment):
                alignment[k].update(range(j1, j2))
    return alignment


def _pronunciations(word: str, expected: list[str], previous_last: str | None, lang: str) -> list[list[str]]:
    """The expected phones, accepted alternatives, and the merged form when two words share a boundary phone."""
    cands = [list(expected)] + [normalize_phones(a, lang) for a in ALTERNATES.get(lang, {}).get(word, [])]
    if previous_last is not None and len(expected) > 1 and expected[0] == previous_last:
        cands.append(list(expected[1:]))
    return cands


def _peak(lp, groups, phone, region) -> float:
    ids = groups.get(phone)
    a, b = region
    if not ids or b <= a:
        return 0.0
    return float(np.exp(lp[a:b][:, ids].max()))


def _phone_reports(candidate, actual, lp, groups, word_region, per_phone_region, lang) -> list[dict]:
    """One report per expected phone: what was heard there and how sure we are it is wrong (0 for a correct phone)."""
    local = _align(candidate, actual)
    last = len(candidate) - 1
    reports = []
    for k, expected in enumerate(candidate):
        heard = [actual[j] for j in sorted(local[k])]
        if heard == [expected]:
            confidence = 0.0
        elif expected in heard:
            confidence = FINAL_EXTRA_COST if k == last else 1.0
        else:
            if heard:
                cost = max(substitution_cost(expected, h, lang, NEAR_COST) for h in heard)
            else:
                cost = FINAL_DELETION_COST if k == last else 1.0
            region = per_phone_region[k] if per_phone_region else word_region
            plausibility = _peak(lp, groups, expected, region)
            confidence = cost * (1.0 - min(1.0, plausibility / PLAUSIBLE_POSTERIOR))
        reports.append({"expected": expected, "heard": " ".join(heard), "error_confidence": round(confidence, 3), "source": "context"})
    return reports


def _context_free_check(samples, reports, model, groups, lang) -> None:
    """Second opinion without context.

    The recogniser's transformer layers restore sounds that are plausible in context ("I sink" is heard
    as "I think"), which hides exactly the substitutions learners make. So each content word is also cut
    out, separated from its neighbours by silence, and decoded alone. That decode is noisier (it
    misfires on short words and at word edges), so it only adds evidence: a phone's error confidence is
    the larger of the two, and the isolated one is weighted down.
    """
    eligible = [r for r in reports if len(r["expected"]) >= CONTEXT_FREE_MIN_PHONES and r["candidate"] == r["expected"]]
    gap = np.zeros(int(CONTEXT_FREE_GAP_S * SAMPLE_RATE), dtype=np.float32)
    batch: list[tuple[dict, int, int]] = []
    pieces: list[np.ndarray] = []
    length = 0

    def flush():
        nonlocal batch, pieces, length
        if not batch:
            return
        lp = model.log_probs(np.concatenate(pieces))
        for rep, core_a, core_b in batch:
            a, b = max(core_a // 320 - 2, 0), min(core_b // 320 + 3, lp.shape[0])
            _merge_isolated(rep, lp[a:b], model, groups, lang)
        batch, pieces, length = [], [], 0

    for rep in eligible:
        w = rep["target"].asr
        a = max(w.start - CONTEXT_FREE_PAD_S, 0.0)
        chunk = samples[int(a * SAMPLE_RATE): int((w.end + CONTEXT_FREE_PAD_S) * SAMPLE_RATE)]
        if length + len(chunk) > SEGMENT_MAX_S * SAMPLE_RATE:
            flush()
        core_a = length + int((w.start - a) * SAMPLE_RATE)
        batch.append((rep, core_a, core_a + int((w.end - w.start) * SAMPLE_RATE)))
        pieces += [chunk, gap]
        length += len(chunk) + len(gap)
    flush()


def _merge_isolated(rep: dict, lp: np.ndarray, model, groups, lang) -> None:
    heard = _heard_phones(lp, model, lang)
    actual = [h["phone"] for h in heard]
    iso = _phone_reports(rep["expected"], actual, lp, groups, (0, lp.shape[0]), None, lang)
    last = len(iso) - 1
    changed = False
    for k, (g, i) in enumerate(zip(rep["phones"], iso)):
        if not i["heard"] and k in (0, last):  # an edge phone missing out of context is usually clipping
            continue
        if g["expected"] in i["heard"].split():  # heard among extra sounds: spill from the word's edges, not a substitution
            continue
        sounds = i["heard"].split()
        if len(sounds) > 1 and k in (0, last):  # keep the sound at the word's inside edge, drop the spill
            i["heard"] = sounds[-1] if k == 0 else sounds[0]
        weighted = i["error_confidence"] * CONTEXT_FREE_WEIGHT
        if weighted > g["error_confidence"]:
            g.update(heard=i["heard"], error_confidence=round(weighted, 3), source="context_free")
            changed = True
    if changed:
        rep["edits"] = sum(p["error_confidence"] for p in rep["phones"])
        rep["context_free_heard"] = " ".join(actual)


# ---- flags, patterns and output -------------------------------------------------------------------

def _set_flag(rep: dict, repeated: set, lang: str) -> None:
    n = len(rep["expected"])
    rep["word_confidence"] = min(1.0, rep["edits"])
    if rep["word_confidence"] < REPORT_CONFIDENCE or n < 2 or rep["target"].replaced:
        rep["flag"] = None
        return
    weak_form = rep["target"].expected in ALTERNATES.get("en", {}) or rep["target"].expected in ALTERNATES.get("fr", {})
    in_pattern = any((p["expected"], p["heard"] or "∅") in repeated for p in rep["phones"] if p["error_confidence"] >= PATTERN_PHONE_CONFIDENCE)
    strong = rep["word_confidence"] >= LIKELY_WORD_CONFIDENCE and n >= 3 and lang == "en"
    rep["flag"] = "likely" if (in_pattern or strong) and not weak_form else "possible"


def _patterns(reports: list[dict]) -> list[dict]:
    """Group confident phone errors by (expected -> heard) across words; repeats are the trustworthy signal."""
    found: dict[tuple[str, str], list[str]] = {}
    for rep in reports:
        if rep["target"].replaced:
            continue
        for p in rep["phones"]:
            if p["error_confidence"] >= PATTERN_PHONE_CONFIDENCE:
                found.setdefault((p["expected"], p["heard"] or "∅"), []).append(rep["target"].word)
    out = []
    for (exp, heard), ws in found.items():
        distinct = sorted(set(ws))
        out.append({"expected": exp, "heard": heard, "occurrences": len(ws), "words_count": len(distinct), "words": distinct,
                    "repeated": len(ws) >= 2})
    out.sort(key=lambda p: (-p["words_count"], -p["occurrences"]))
    return out


def _public(rep: dict, with_phones: bool = False) -> dict:
    t: Target = rep["target"]
    d = {
        "word": t.word, "asr_index": t.asr.index, "start": round(t.asr.start, 2), "end": round(t.asr.end, 2),
        "asr_confidence": None if t.asr.confidence is None else round(t.asr.confidence, 2),
        "expected": " ".join(rep["expected"]), "heard": " ".join(rep["heard"]),
        "error_confidence": round(rep["word_confidence"], 2), "flag": rep["flag"],
    }
    if rep["context_free_heard"]:
        d["context_free_heard"] = rep["context_free_heard"]
    if with_phones or rep["flag"]:
        d["phones"] = [p for p in rep["phones"] if p["error_confidence"] > 0]
    return d
