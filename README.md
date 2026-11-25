# speech-check

A local command-line tool that analyses a short speaking-practice recording (English first, French at lower
reliability) and returns measurements a text-only language model (or a person) can reason about: a verbatim transcript with
fillers and repetitions kept, fluency numbers from word timings, phone-level pronunciation hints, and
per-sentence pitch variation. Output is JSON plus a short readable summary. Every signal carries an honest
reliability note.

Nothing leaves the machine: after the one-time model download (`speech-check setup`) every command runs with
the Hugging Face hub forced offline. CPU only; a 90 s clip takes about 1 to 1.5 minutes on 4 cores.

## Install

```sh
uv tool install git+https://github.com/soffits/speech-check
speech-check setup     # downloads the models (about 4 GB on disk) into $SPEECH_CHECK_HOME
speech-check doctor    # checks ffmpeg, espeak-ng, models
```

From a clone, `uv sync` gives the pinned environment (CPU-only torch) and `uv run speech-check ...` runs it.

Needs `ffmpeg` on PATH. espeak-ng ships inside a Python wheel, no system package. Models and their converted
copies live under `$SPEECH_CHECK_HOME` (default `~/.cache/speech-check`).

## Use

```sh
speech-check analyze clip.ogg                          # readable summary
speech-check analyze clip.ogg --format json            # full JSON on stdout
speech-check analyze clip.ogg --json-out result.json   # summary on stdout, JSON in a file
speech-check analyze clip.ogg --lang fr                # French (pronunciation part is uncalibrated)
speech-check analyze clip.ogg --reference "The text that was meant to be read."   # read-aloud mode
speech-check analyze clip.ogg --asr-model small        # faster, less accurate (small|medium|turbo|large)
speech-check analyze clip.ogg --no-pronunciation       # skip the slow phone layer
speech-check schema                                    # JSON Schema of the result
```

Any file ffmpeg can read works (for example OGG/Opus voice messages). Exit code 1 with a message on stderr
when the file cannot be read or has no speech.

## What it measures

| Signal | How | Good for | Not good for |
|---|---|---|---|
| Transcript | CrisperWhisper 2.0 (medium, int8), verbatim: keeps um/uh, repeats, cut-off words, with word times and confidence | what was said, filler and repetition counts | seeing mispronunciations (the model writes the intended word); counts are lower bounds |
| Fluency | speech rate, articulation rate, mean length of run, pauses (>= 0.25 s, long >= 0.6 s) by position, fillers, repetitions, cut-offs/self-corrections, from word timings | the most dependable signal; compare clips of the same task | pause position relies on ASR punctuation (approximate); no calibrated "good" range |
| Pronunciation | espeak-ng expected phones vs wav2vec2 phone recogniser, CTC forced alignment for plausibility, plus an out-of-context decode per content word | sounds that go wrong again and again ("repeated" patterns) | single-word verdicts, stress, rhythm, accent overall; French is uncalibrated |
| Intonation | Praat pitch (parselmouth), two-pass limits, octave-jump cleaning, semitones per sentence | flat vs varied delivery, relative changes over time | absolute pitch values; final rise/fall is unstable |

Details, definitions and field list: [docs/schema.md](docs/schema.md). What was tested and where it fails:
[docs/evaluation.md](docs/evaluation.md). Licences of every component: [docs/licences.md](docs/licences.md).

## How pronunciation is judged

For each word, the phones espeak-ng expects (from the transcript, or from `--reference` in read-aloud mode)
are compared with the phones the recogniser hears. Each wrong phone gets an error confidence that falls when
the expected phone was plausible in the aligned audio. Two decodes are used: in context, and a second one on
each content word cut out alone, because the recogniser tends to restore sounds that fit the context ("I sink"
is heard as "I think"). The out-of-context decode is noisier, so it is weighted down and only adds evidence.
A word is "likely" only when the same wrong sound shows up two or more times, or (English) one error is very
confident; everything else is "possible". Published read-English benchmarks of this kind of method show
roughly one in five flagged words was really bad, so treat flags as things to listen for.

## Development

```sh
uv run pytest                      # unit tests (no models needed)
SPEECH_CHECK_HOME=... SPEECH_CHECK_VOICES=<piper voice dir> uv run pytest -m integration
uv run python tests/make_clips.py OUT_DIR VOICES_DIR   # synthetic test clips (Piper, dev dependency)
```

Copyright (C) 2026 Sakina. Licence: AGPL-3.0-or-later, see [LICENSE](LICENSE). That covers this code only. The models are separate
downloads with their own terms: the CrisperWhisper 2.0 speech model is **non-commercial research only**, which limits what you can do with
the tool's transcripts, see [docs/licences.md](docs/licences.md) and [NOTICE.md](NOTICE.md).
