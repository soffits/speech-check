import numpy as np

from speech_check import pronunciation as pr


def test_align_maps_insertions_to_previous_phone():
    alignment = pr._align(["a", "b"], ["a", "x", "b"])
    assert alignment[0] == {0, 1} and alignment[1] == {2}


def test_read_aloud_reference_comparison(mk):
    words = mk([("Last", 0, .3), ("weekend", .3, .8), ("I", .8, .9), ("went", .9, 1.2), ("far", 1.2, 1.5)])
    targets, rep = pr._plan_targets(words, "Last weekend I visited my friend")
    assert rep["words_in_reference"] == 6
    assert rep["matched"] == 3
    assert rep["substituted"] == [{"expected": "visited", "said": "went"}, {"expected": "my", "said": "far"}]
    assert rep["omitted"] == ["friend"]
    assert rep["inserted"] == []
    assert [t.word for t in targets if t.replaced] == ["visited", "my"]


def test_free_speech_targets_skip_fillers(mk):
    words = mk([("[UM]", 0, .3), ("hello", .3, .6), ("th-", .6, .7), ("there.", .7, 1.0)])
    targets, rep = pr._plan_targets(words, None)
    assert [t.word for t in targets] == ["hello", "there"] and rep is None


def test_segments_cut_at_pauses(mk):
    spec, t = [], 0.0
    for i in range(30):  # 30 words of 0.5 s with a 0.4 s pause after each
        spec.append((f"w{i}", t, t + 0.5))
        t += 0.9
    segs = pr._segments(mk(spec))
    assert len(segs) >= 2
    assert all(seg[-1].end - seg[0].start <= pr.SEGMENT_MAX_S + 1 for seg in segs)
    assert sum(len(s) for s in segs) == 30


def report(word, expected, phones, replaced=False, asr=None):
    from speech_check.words import make_word
    t = pr.Target(word, word, make_word(0, word, 0, 1, 0.9), replaced=replaced)
    edits = sum(p["error_confidence"] for p in phones)
    return {"target": t, "expected": expected, "phones": phones, "edits": edits, "context_free_heard": None,
            "heard": [], "candidate": expected}


def phone(e, h, c):
    return {"expected": e, "heard": h, "error_confidence": c, "source": "context"}


def test_flags_repeated_pattern_is_likely_single_weak_is_possible():
    a = report("think", list("θɪŋk"), [phone("θ", "s", 0.76)])
    b = report("thank", list("θæŋk"), [phone("θ", "s", 0.7)])
    c = report("lunch", list("lʌntʃ"), [phone("ʌ", "ə", 0.55)])
    reports = [a, b, c]
    patterns = pr._patterns(reports)
    assert patterns[0]["expected"] == "θ" and patterns[0]["repeated"] and patterns[0]["occurrences"] == 2
    repeated = {(p["expected"], p["heard"]) for p in patterns if p["repeated"]}
    for r in reports:
        pr._set_flag(r, repeated, "en")
    assert [r["flag"] for r in reports] == ["likely", "likely", "possible"]


def test_strong_single_error_likely_in_english_only():
    r = report("berry", list("vɛɹi"), [phone("v", "b", 0.95)])
    pr._set_flag(r, set(), "en")
    assert r["flag"] == "likely"
    r = report("voir", list("vwaʁ"), [phone("w", "", 0.95)])
    pr._set_flag(r, set(), "fr")
    assert r["flag"] == "possible"


def test_replaced_word_and_weak_form_not_flagged_as_likely():
    r = report("lunch", list("lʌntʃ"), [phone("l", "d", 1.0)], replaced=True)
    pr._set_flag(r, set(), "en")
    assert r["flag"] is None
    r = report("the", list("ðə"), [phone("ð", "z", 1.0)])
    r["target"].expected = "the"
    pr._set_flag(r, set(), "en")
    assert r["flag"] == "possible"


def test_peak_posterior():
    lp = np.log(np.full((5, 3), 0.1))
    lp[2, 1] = np.log(0.7)
    assert abs(pr._peak(lp, {"x": [1]}, "x", (0, 5)) - 0.7) < 1e-6
    assert pr._peak(lp, {}, "x", (0, 5)) == 0.0
