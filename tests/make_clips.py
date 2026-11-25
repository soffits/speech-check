"""Build synthetic practice clips with Piper (dev dependency) for manual checks.

Usage: python tests/make_clips.py OUT_DIR VOICES_DIR

Writes clean.wav, planted.wav and french.wav (+ .ogg Opus copies) and a clips.json that lists
what was planted, so a run's output can be compared against it.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from piper import PiperVoice

SR = 16000


def synth(voice, text, speed=1.0):
    from piper import SynthesisConfig
    cfg = SynthesisConfig(length_scale=1.0 / speed)
    chunks = [c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)]
    audio = np.concatenate(chunks)
    if voice.config.sample_rate != SR:
        n = int(len(audio) * SR / voice.config.sample_rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def build(voice, parts):
    out = []
    for text, pause in parts:
        if text:
            out.append(synth(voice, text))
        out.append(np.zeros(int(SR * pause), dtype=np.float32))
    return np.concatenate(out)


CLEAN = [
    ("Last weekend I visited my friend in the countryside.", 0.4),
    ("We went for a long walk by the river, and then we cooked dinner together.", 0.4),
    ("I think it was a very good way to relax after a busy week.", 0.4),
    ("Next time, I would like to invite more people, because very few of them have visited.", 0.3),
    ("Thank you for listening.", 0.2),
]

PLANTED = [
    ("Last weekend I visited my friend in the countryside.", 0.4),
    # planted: "sink" for "think", "berry" for "very"
    ("I sink it was a berry good way to relax after a busy week.", 0.4),
    # filled pauses and a long pause inside a clause
    ("Um", 0.5), ("we went for a long", 1.4), ("walk by the river.", 0.3),
    ("Uh", 0.4), ("and then we cooked dinner together, it was berry relaxing.", 0.9),
    # repetitions
    ("Next time I", 0.15), ("I would like to", 0.12), ("like to invite more people,", 0.2),
    ("because I sink berry few of them have visited.", 0.3),
    ("Thank you for listening.", 0.2),
]

FRENCH = [
    ("Le week-end dernier, je suis allée voir une amie à la campagne.", 0.4),
    ("Nous avons fait une longue promenade au bord de la rivière.", 1.3),
    ("Ensuite, nous avons préparé le dîner ensemble.", 0.3),
]

CLIPS = {
    "clean": ("en_US-amy-medium", CLEAN),
    "planted": ("en_US-amy-medium", PLANTED),
    "french": ("fr_FR-siwis-medium", FRENCH),
}


def main(out_dir, voices_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, (voice_name, parts) in CLIPS.items():
        voice = PiperVoice.load(Path(voices_dir) / f"{voice_name}.onnx")
        audio = build(voice, parts)
        wav = out / f"{name}.wav"
        sf.write(wav, audio, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", "-b:a", "24k",
                        str(out / f"{name}.ogg")], check=True)
        print(name, round(len(audio) / SR, 1), "s")
    (out / "clips.json").write_text(json.dumps({k: [p[0] for p in v[1]] for k, v in CLIPS.items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
