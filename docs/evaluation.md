# What was tested, and what failed

Everything here was measured on a 4-core CPU on 2026-11-17. Treat the numbers as
indications: the clips are few, and synthetic speech is cleaner than a person.

## Speed and memory

| Clip | Total | ASR (medium, int8) | Phone layer | Peak RSS |
|---|---|---|---|---|
| 26 s synthetic | 25 s | 5 s plus model load | 19 s | |
| 89 s synthetic (3 clips joined, Opus 24 kbit/s) | 72-76 s | 35-39 s | 32-40 s | 3.0 GB |

ASR alone for medium was 4.6 s for 26 s of audio on an idle machine (small: 1.7 s). The phone layer is the
largest cost: two wav2vec2 passes (in context, then each content word alone). Dynamic int8 quantisation of
wav2vec2 was 1.8x faster with 99% frame agreement but uses a deprecated torch API, so it is not used.

## Synthetic clips with planted problems (Piper voice, Opus)

Planted: "sink" for "think", "berry" for "very", a filler "um" and "uh", a 1.4 s pause inside a clause, "I I",
"like to like to", repeated several times where noted.

| Planted | Result |
|---|---|
| fillers `um`, `uh` | both found (`[UM]`, `[UH]`), with times |
| 1.4 s pause inside a clause | found as a 2.0 s long pause "within clause" (the TTS adds silence of its own) |
| "like to like to" | found as a repetition |
| "I I" | found in one run; missed in another where the synthesiser ran the two words together |
| pauses at sentence ends | found, labelled between sentences |
| "berry" for "very" (3 times) | `/v/` heard as `/b/` reported as a repeated pattern (2 of 3 instances; the third was heard as /v/) |
| "sink" for "think" (3 times) | `/θ/` heard as `/s/` reported as a repeated pattern (2 of 3). In the first runs without the out-of-context decode it was a complete miss: the in-context recogniser heard /θ/ |
| clean control clip, same voice | no flags at all |

Misses and caveats: the ASR transcript itself always repaired "sink" and "berry" to "think" and "very", so
the transcript alone shows nothing. Without the out-of-context decode the phone layer missed both errors. With
it, a single instance is only "possible"; the pattern needs two.

French (Piper voice, 12 s): transcript, pauses and pitch fine; phone hints appeared on a clean clip
("voir", "longue", "bord") and are meaningless without calibration, hence "very low" reliability and no
single-word "likely" for French.

## Real learner speech (speechocean762, CC BY 4.0, 5 adult utterances, read-aloud mode; Zhang et al., 2021, *speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment*, Interspeech)

| Utterance | Human accuracy | Result |
|---|---|---|
| "follow the link below to register" | 8 | caught "register" /ɛ/ heard as /ɪ/, the exact error the annotators marked; also flagged "link" and "below" as possible (not annotated) |
| "it was a proof of his friendship" | 9 | missed "proof" /uw/ to /ow/ (annotated) |
| three utterances rated 10/10 | 10 | one false "likely" ("ladder"), two false "possible" ("step", "good") |

That is one hit, one miss and three false alarms on five clips: no statistics, but consistent with the one-in-five
precision other benchmarks report for this approach. Word-flag precision is low; only repeated patterns are
worth stating to a learner as a likely issue, and even those should be phrased as "listen for".

## Not measured

Real voice messages from a phone, noisy rooms, accents other than the five speakers above,
French learner speech, pitch accuracy against a reference tracker (SwiftF0 was not compared), CrisperWhisper
large and turbo speed, faster-whisper as an alternative engine.
