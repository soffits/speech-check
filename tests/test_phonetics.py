from speech_check.phonetics import normalize_phone, normalize_phones, phones_for_words, substitution_cost, syllable_count


def test_normalisation():
    assert normalize_phone("ɑː", "en") == "ɑ"
    assert normalize_phone("ɔ", "en") == "ɑ"
    assert normalize_phone("ɛ", "fr") == "e"
    assert normalize_phones(["ɚ", "ɹ", "d", "d"], "en") == ["ɚ", "d"]


def test_substitution_cost():
    assert substitution_cost("θ", "s", "en", 0.5) == 0.5
    assert substitution_cost("v", "b", "en", 0.5) == 1.0


def test_espeak_english_and_french():
    assert phones_for_words(["think"], "en")[0] == ["θ", "ɪ", "ŋ", "k"]
    fr = phones_for_words(["je", "suis"], "fr")
    assert len(fr) == 2 and fr[0][0] == "ʒ"


def test_syllables():
    assert syllable_count(["θ", "ɪ", "ŋ", "k"], "think") == 1
    assert syllable_count(["v", "ɛ", "ɹ", "i"], "very") == 2
    assert syllable_count([], "banana") == 3
