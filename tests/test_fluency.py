from speech_check import fluency


def speech(mk):
    # "I went to the store. [UM] Then I I went home" with controlled gaps
    return mk([
        ("I", 0.0, 0.2), ("went", 0.2, 0.5), ("to", 0.5, 0.6), ("the", 0.6, 0.7), ("store.", 0.7, 1.2),
        ("[UM]", 2.4, 2.7),
        ("Then", 2.8, 3.1), ("I", 3.1, 3.2), ("I", 3.25, 3.35), ("went", 3.4, 3.7), ("th-", 3.7, 3.8), ("home.", 3.9, 4.4),
    ])


def test_rates_and_counts(mk):
    f = fluency.analyze(speech(mk), duration=5.0, lang="en")
    assert f["available"]
    assert f["words"] == 10  # the cut-off 'th-' is not counted as a word
    assert f["fillers"]["count"] == 1
    assert f["speaking_span_s"] == 4.4
    assert f["speech_rate_wpm"] == round(10 / (4.4 / 60), 1)
    assert 0 < f["phonation_time_ratio"] < 1


def test_long_pause_and_position(mk):
    f = fluency.analyze(speech(mk), duration=5.0, lang="en")
    long = f["pauses"]["long"]
    assert len(long) == 1
    assert long[0]["position"] == "between_sentences"
    assert long[0]["duration"] == 1.2
    assert "store." in long[0]["context"]
    assert f["pauses"]["long_by_position"] == {"between_sentences": 1}


def test_pause_after_function_word_is_within_clause(mk):
    words = mk([("We", 0, 0.2), ("went", 0.2, 0.5), ("for", 0.5, 0.7), ("a.", 0.7, 0.8), ("walk", 2.0, 2.4)])
    f = fluency.analyze(words, 3.0, "en")
    assert f["pauses"]["long"][0]["position"] == "within_clause"


def test_repetitions_and_cutoffs(mk):
    f = fluency.analyze(speech(mk), 5.0, "en")
    reps = f["repetitions"]["items"]
    assert [(r["text"], r["times"]) for r in reps] == [("i", 2)]
    assert f["self_corrections"]["count"] == 1
    assert f["self_corrections"]["items"][0]["type"] in ("cut_off", "restart")


def test_phrase_repetition_counts_once(mk):
    words = mk([("I", 0, .2), ("would", .2, .4), ("like", .4, .6), ("to", .6, .7), ("like", .7, .9), ("to", .9, 1.0),
                ("invite", 1.0, 1.4), ("them", 1.4, 1.6)])
    reps = fluency.analyze(words, 2.0, "en")["repetitions"]["items"]
    assert [(r["text"], r["times"]) for r in reps] == [("like to", 2)]


def test_repair_phrase_french(mk):
    words = mk([("Je", 0, .2), ("veux", .2, .4), ("dire", .4, .6), ("non", .6, .9)])
    assert fluency.analyze(words, 1.0, "fr")["self_corrections"]["items"][0]["type"] == "repair_phrase"


def test_no_words():
    assert fluency.analyze([], 1.0, "en")["available"] is False
