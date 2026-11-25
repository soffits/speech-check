import subprocess

import numpy as np
import pytest
import soundfile as sf

from speech_check import audio, intonation

SR = 16000


def tone(freqs_hz, seconds_each=0.5):
    """Voiced-like signal: harmonic-rich pulses at the given pitches, one after another."""
    out = []
    for f in freqs_hz:
        t = np.arange(int(SR * seconds_each)) / SR
        out.append(sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8)))
    x = np.concatenate(out).astype(np.float32)
    return 0.3 * x / np.abs(x).max()


def test_decode_any_ffmpeg_format(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, tone([200]), SR)
    ogg = tmp_path / "a.ogg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libopus", str(ogg)], check=True)
    clip = audio.load(str(ogg))
    assert abs(clip.duration - 0.5) < 0.1 and clip.samples.dtype == np.float32
    assert clip.source.get("codec") == "opus"


def test_too_short_audio_rejected(tmp_path):
    wav = tmp_path / "s.wav"
    sf.write(wav, np.zeros(1000, dtype=np.float32), SR)
    with pytest.raises(audio.AudioError):
        audio.load(str(wav))


def test_quality_warnings():
    q = audio.quality(0.001 * np.random.default_rng(0).standard_normal(SR * 3).astype(np.float32))
    assert any("signal-to-noise" in w or "quiet" in w for w in audio.quality_warnings(3.0, q))
    assert any("shorter than 10 s" in w for w in audio.quality_warnings(3.0, q))


def sentence_words(mk, end):
    return mk([("one", 0.0, end / 2), ("two.", end / 2, end)])


def test_flat_vs_varied_pitch(mk):
    flat = tone([200] * 6)
    varied = tone([150, 220, 130, 260, 170, 240])
    r_flat = intonation.analyze(flat, sentence_words(mk, 3.0))
    r_var = intonation.analyze(varied, sentence_words(mk, 3.0))
    assert r_flat["available"] and r_var["available"]
    assert r_var["sentences"][0]["range_st"] > r_flat["sentences"][0]["range_st"] + 4
    assert r_flat["sentences"][0]["flat"] is True
    assert r_var["sentences"][0]["flat"] is False


def test_octave_cleaning_folds_jumps():
    st = np.zeros(120)
    st[50:60] = 12.0  # a one-octave error in the middle of a steady voice
    cleaned = intonation._clean(st)
    assert np.nanmax(np.abs(cleaned)) < 3
