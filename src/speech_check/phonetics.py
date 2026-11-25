"""Expected phones from espeak-ng, and the phone normalisation shared with the recogniser's label set.

The normalisation tables, near-phone groups and weak-form alternatives are adapted from OpenPronounce
(MIT, https://github.com/OpenPronounce/OpenPronounce, openpronounce/phones.py); see NOTICE.md.
"""
from __future__ import annotations

import re
from functools import lru_cache

ESPEAK_VOICE = {"en": "en-us", "fr": "fr-fr"}

_TOKEN_ALIASES = {"th": "t", "kh": "k", "ph": "p", "tʰ": "t", "kʰ": "k", "pʰ": "p",
                  "ai": "aɪ", "ei": "eɪ", "au": "aʊ", "ou": "oʊ"}
_TONE = re.compile(r"\d")

# Phones espeak and the recogniser use interchangeably, or that no learner should be penalised for.
_PHONE_MAP = {
    "en": {"ᵻ": "ɪ", "ɐ": "ə", "ɔ": "ɑ", "ɔɹ": "oɹ", "ɜ": "ɚ", "ɾ": "t", "ɫ": "l"},
    "fr": {"ɛ": "e", "œ": "ø", "ə": "ø", "ɔ": "o"},  # mid-vowel pairs that follow position, not meaning
}

# Close phones: substituting one for another is a minor error (voicing, tense/lax, typical L2 approximations).
NEAR_PHONES = {
    "fr": [{"e", "i"}, {"ø", "y"}, {"ø", "e"}, {"o", "u"}, {"ɑ̃", "ɔ̃"}, {"ɛ̃", "ɑ̃"}, {"ʁ", "r", "ɹ"},
           {"b", "p"}, {"d", "t"}, {"ɡ", "k"}, {"z", "s"}, {"v", "f"}, {"ʒ", "ʃ"}],
    "en": [{"ɪ", "i"}, {"ʊ", "u"}, {"ɛ", "ə"}, {"ɪ", "ə"}, {"θ", "ð"}, {"ɛ", "æ"}, {"ʌ", "ɑ", "ə", "a"}, {"æ", "a"}, {"ɛ", "eɪ"},
           {"ð", "z", "d"}, {"θ", "s", "t"}, {"b", "p"}, {"d", "t"}, {"ɡ", "k"}, {"z", "s"},
           {"v", "f"}, {"ʒ", "ʃ"}, {"dʒ", "tʃ"}, {"n", "ŋ"}, {"h", "x"}, {"ɹ", "r"}, {"v", "w"}],
}

# Function words with several accepted pronunciations (phones already normalised).
ALTERNATES = {
    "en": {"a": [["ə"], ["eɪ"]], "an": [["ə", "n"], ["æ", "n"]], "the": [["ð", "ə"], ["ð", "i"], ["ð", "ɪ"]],
           "to": [["t", "ə"], ["t", "u"]], "of": [["ʌ", "v"], ["ə", "v"]],
           "and": [["æ", "n", "d"], ["ə", "n", "d"], ["ə", "n"]], "for": [["f", "ɑ", "ɹ"], ["f", "ɚ"]],
           "you": [["j", "u"], ["j", "ə"]], "are": [["ɑ", "ɹ"], ["ɚ"]], "was": [["w", "ʌ", "z"], ["w", "ə", "z"]],
           "that": [["ð", "æ", "t"], ["ð", "ə", "t"]], "can": [["k", "æ", "n"], ["k", "ə", "n"]],
           "have": [["h", "æ", "v"], ["h", "ə", "v"]], "or": [["ɑ", "ɹ"], ["ɚ"]]},
    # schwa elision in French function words ("je suis" -> /ʒsɥi/)
    "fr": {"je": [["ʒ"]], "le": [["l"]], "de": [["d"]], "ne": [["n"]], "ce": [["s"]], "se": [["s"]],
           "me": [["m"]], "te": [["t"]], "que": [["k"]]},
}

_VOWEL_START = set("aeiouɛɪɔʊʌæɑəɚɜɐøœyɨʉɒɯɘɵɤ")
_VOWEL_GROUPS = re.compile(r"[aeiouyàâäéèêëîïôöùûü]+", re.IGNORECASE)


def normalize_phone(phone: str, lang: str) -> str:
    phone = _TONE.sub("", phone.replace("ː", ""))
    phone = _TOKEN_ALIASES.get(phone, phone)
    return _PHONE_MAP.get(lang, {}).get(phone, phone)


def normalize_phones(phones: list[str], lang: str) -> list[str]:
    """Normalise and collapse immediate repeats ("ɚ ɹ" -> "ɚ", "d d" -> "d")."""
    out: list[str] = []
    for p in phones:
        p = normalize_phone(p, lang)
        if not p or (out and (out[-1] == p or (out[-1] == "ɚ" and p == "ɹ"))):
            continue
        out.append(p)
    return out


def substitution_cost(expected: str, heard: str, lang: str, near_cost: float) -> float:
    for group in NEAR_PHONES.get(lang, []):
        if expected in group and heard in group:
            return near_cost
    return 1.0


@lru_cache(maxsize=2)
def _backend(lang: str):
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return EspeakBackend(ESPEAK_VOICE[lang], preserve_punctuation=False, with_stress=False, language_switch="remove-flags")


def phones_for_words(words: list[str], lang: str) -> list[list[str]]:
    """Raw espeak phones for each word (not normalised). The words are phonemised together so
    connected-speech forms ("a" before a consonant) come out right; a mismatch falls back to word by word."""
    from phonemizer.separator import Separator

    if not words:
        return []
    backend = _backend(lang)
    joined = backend.phonemize([" ".join(words)], separator=Separator(phone=" ", word="|", syllable=""), strip=True)[0]
    groups = [g.split() for g in joined.split("|")]
    if len(groups) == len(words):
        return groups
    singles = backend.phonemize(words, separator=Separator(phone=" ", word="", syllable=""), strip=True)
    return [g.split() for g in singles]


def syllable_count(raw_phones: list[str], word: str) -> int:
    """Syllables of a word: its vowel phones, or vowel letter groups when espeak gave no phones."""
    if raw_phones:
        return max(1, sum(1 for p in raw_phones if p and p[0] in _VOWEL_START))
    return max(1, len(_VOWEL_GROUPS.findall(word))) if word else 0
