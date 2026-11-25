from speech_check.words import classify, normalize, sentences, make_word


def test_classify_tokens():
    assert classify("[UM]") == "filler"
    assert classify("[uh]") == "filler"
    assert classify("um,") == "filler"
    assert classify("euh") == "filler"
    assert classify("[laughter]") == "sound"
    assert classify("th-") == "cutoff"
    assert classify("Thursday.") == "word"
    assert classify("-") == "word"  # a lone dash is not a cut-off word


def test_normalize_keeps_apostrophes_and_accents():
    assert normalize("Don’t,") == "don't"
    assert normalize("été.") == "été"
    assert normalize("week-end") == "week-end"


def test_boundary_from_punctuation():
    assert make_word(0, "week.", 0, 1, 1).boundary == "sentence"
    assert make_word(0, "week,", 0, 1, 1).boundary == "clause"
    assert make_word(0, "week", 0, 1, 1).boundary is None
    assert make_word(0, "week?\"", 0, 1, 1).boundary == "sentence"


def test_sentences_split_and_skip_sounds(mk):
    words = mk([("Hi.", 0, 1), ("[laughter]", 1, 2), ("How", 2, 3), ("are", 3, 4), ("you?", 4, 5)])
    parts = sentences(words)
    assert [[w.text for w in s] for s in parts] == [["Hi."], ["How", "are", "you?"]]
