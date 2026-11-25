"""How far each signal can be trusted. These notes travel with every result."""
from __future__ import annotations


def notes(lang: str, mode: str, snr_db: float | None) -> dict:
    noisy = snr_db is not None and snr_db < 15
    transcript = {
        "level": "medium" if noisy else "high",
        "note": ("Words and timings are usually right on clear speech. The recogniser tends to write the intended word "
                 "even when a sound was wrong (it can write 'think' for 'sink'), so the transcript cannot show "
                 "mispronunciations. Fillers, repetitions and cut-off words are kept by the verbatim model but not every one "
                 "is caught, so their counts are lower bounds."),
    }
    fluency = {
        "level": "medium" if noisy else "high",
        "note": ("Speech rate, articulation rate and pause timings follow the ASR word timings (about 30-40 ms error on clean "
                 "speech) and are the most dependable numbers here. Pause position (within clause vs between sentences) "
                 "relies on the ASR's punctuation and is approximate: read the quoted context. Compare clips of the same task type; "
                 "there is no calibrated 'good' range for this learner."),
    }
    if lang == "fr":
        pron = {
            "level": "very_low",
            "note": ("French is supported by the same phone model but nothing here is calibrated for it: no scored French "
                     "learner data exists to check against. Treat any phone hint as a suggestion to listen for, and trust only "
                     "sounds flagged in several words."),
        }
    else:
        pron = {
            "level": "low",
            "note": ("Phone hints come from a general phone recogniser, not a trained pronunciation scorer. On read English "
                     "the closest published benchmark (OpenPronounce, 500 speechocean762 utterances) found about 70% of the "
                     "words humans rejected but only about 1 in 5 flagged words was actually bad; free speech is uncalibrated. "
                     "Only sounds wrong in several words ('repeated' patterns) are reported as likely; single-word flags are "
                     "'possible' and need other evidence. The phone layer cannot judge stress, rhythm or accent as a whole."),
        }
    if mode == "free_speech":
        pron["note"] += (" In free speech the expected sounds come from the transcript, so a sound the ASR silently corrected "
                         "can still be caught here, but a word the ASR misheard is checked against the wrong word.")
    inton = {
        "level": "low" if noisy else "medium",
        "note": ("Pitch range and variation in semitones are meaningful relative to the same speaker over time. Absolute "
                 "values depend on the microphone, phone codec and sentence length; sentences under 0.6 s of voiced speech "
                 "are skipped. Final rise/fall is the least stable measure. No calibrated 'flat' cut-off exists; the flat "
                 "flag uses a rough rule of thumb (range under 5 semitones and sd under 1.5)."),
    }
    if lang == "fr":
        inton["note"] += " French phrase-final intonation follows different patterns from English, so do not apply English expectations."
    return {"transcript": transcript, "fluency": fluency, "pronunciation": pron, "intonation": inton}
