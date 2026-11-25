# Result schema (version 1)

`speech-check analyze --format json` prints one JSON object; `speech-check schema` prints its JSON Schema
(`src/speech_check/result.schema.json`, checked against real output in the tests).

**Stability:** within `schema_version: 1` fields are only ever added, never renamed, removed or re-typed.
Readers must ignore unknown fields. A breaking change bumps `schema_version`.

Times are seconds from the start of the audio. Pitch is in semitones. Confidences are 0 to 1.

## Top level

| Field | Meaning |
|---|---|
| `schema_version`, `tool` | `1`; `{name, version}` |
| `input` | `file` (name only), `duration_s`, `language` (`en`/`fr`), `mode` (`free_speech`/`read_aloud`), `reference_text`, `source` (codec etc.) |
| `recording` | level, noise floor, `snr_db_estimate`, `clipped_fraction`, `warnings[]` (short clip, noise, clipping) |
| `models` | which ASR / phone / pitch components ran |
| `reliability` | per signal `{level: high|medium|low|very_low, note}`. Read the note before using a number |
| `transcript` | `text`, `words[]`, `sentences[]` |
| `fluency` | see below |
| `pronunciation` | absent with `--no-pronunciation` |
| `intonation` | absent with `--no-intonation` |
| `timing_s` | seconds per stage and `total` |
| `summary_text` | the readable summary |

## transcript.words[]

`i` (index), `text` as the ASR wrote it (punctuation kept; fillers look like `[UM]`), `start`, `end`,
`confidence` (ASR word probability), `kind`: `word`, `filler`, `cutoff` (a word cut off, `th-`), `sound`
(laughter and similar).

## fluency

- `speech_rate_wpm`: words per minute over the speaking span, pauses included. `speech_rate_syll_per_s` likewise in syllables (counted from espeak vowel phones).
- `articulation_rate_syll_per_s`: syllables per second with silent pauses removed. `phonation_time_ratio`: share of the span spent speaking.
- `mean_length_of_run_*`: average stretch of speech between pauses of at least `pauses.threshold_s` (0.25 s).
- `pauses`: `count`, `per_minute`, `mean_duration_s`, `longest_s`, `by_position`, and for pauses of at least 0.6 s `long_count`, `long_by_position` and `long[]` with `start`, `end`, `duration`, `position`, `after_word`, `before_word`, `context` (words around the pause, `|` marks it).
  `position` is `within_clause`, `between_clauses`, `between_sentences` or `after_filler`. It follows the ASR's punctuation, except that a pause after a word that cannot end a clause (a, the, to, and...) counts as within a clause. Approximate: read `context`.
- `fillers`, `repetitions` (immediate repeats of 1 to 3 words, fillers skipped), `self_corrections` (`restart`: a cut-off word then the same word again; `cut_off`; `repair_phrase`: "I mean", "sorry"...). Counts are lower bounds.

## pronunciation

- `words[]`: every assessed word with `expected` and `heard` phones (IPA, space separated), `asr_index` (into `transcript.words`), `error_confidence`, `flag` (`likely`, `possible` or `null`); flagged words also have `phones[]` (`expected`, `heard`, `error_confidence`, `source`: `context` or `context_free`) and possibly `context_free_heard`.
- `flagged[]`: the words with a flag, with phone detail.
- `patterns[]`: wrong sounds grouped as `expected` to `heard`, with `occurrences`, `words`, and `repeated` (two or more occurrences, the most trustworthy signal).
- `overall`: counts and `phone_error_rate` (includes recogniser noise: clean text-to-speech measured 0.06 to 0.09).
- `reference` (read-aloud mode): `word_accuracy`, `substituted` (said another word, not counted as a sound error), `omitted`, `inserted`.

Flag rules: a word is reported at summed phone error confidence of 0.5 or more. `likely` needs a phone that is
part of a repeated pattern, or (English only) confidence 0.9 or more on a word of 3+ phones; weak-form function
words are never `likely`; one-phone words and replaced words are never flagged. Thresholds are uncalibrated.

## intonation

Per sentence (split at the ASR's sentence punctuation) with at least 0.6 s of voiced speech: `range_st` (5th to
95th percentile), `sd_st`, `median_hz`, `flat` (range under 5 st and sd under 1.5, a rough rule of thumb),
`final_movement_st` and `final_contour` (`rising`, `falling`, `level`; least stable). `summary` holds medians.
