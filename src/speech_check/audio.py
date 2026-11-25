"""Decoding any ffmpeg-readable file to 16 kHz mono float samples, plus recording-quality checks."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000


class AudioError(RuntimeError):
    pass


@dataclass
class Audio:
    samples: np.ndarray  # float32 mono, SAMPLE_RATE
    source: dict  # codec / sample rate / channels of the input file, when ffprobe is available

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


def load(path: str) -> Audio:
    if shutil.which("ffmpeg") is None:
        raise AudioError("ffmpeg not found on PATH")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not read {path!r}: {proc.stderr.decode(errors='replace').strip()[:300]}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(samples) < SAMPLE_RATE // 2:
        raise AudioError("audio is shorter than half a second")
    return Audio(samples=samples, source=_probe(path))


def _probe(path: str) -> dict:
    if shutil.which("ffprobe") is None:
        return {}
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels",
         "-of", "json", path],
        capture_output=True,
    )
    try:
        stream = json.loads(proc.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return {}
    return {"codec": stream.get("codec_name"), "sample_rate": int(stream.get("sample_rate", 0)) or None,
            "channels": stream.get("channels")}


def quality(samples: np.ndarray) -> dict:
    """Level, clipping and a rough signal-to-noise estimate from 20 ms frame energies."""
    frame = SAMPLE_RATE // 50
    n = len(samples) // frame
    rms = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    noise, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    peak = float(np.max(np.abs(samples)))
    return {
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
        "speech_level_dbfs": round(speech, 1),
        "noise_floor_dbfs": round(noise, 1),
        "snr_db_estimate": round(speech - noise, 1),
        "clipped_fraction": round(float(np.mean(np.abs(samples) >= 0.999)), 5),
    }


def quality_warnings(duration: float, q: dict) -> list[str]:
    out = []
    if duration < 10:
        out.append("clip shorter than 10 s: rates, pause statistics and pitch ranges are unstable")
    if duration > 180:
        out.append("clip longer than 3 minutes: analysis will be slow")
    if q["snr_db_estimate"] < 15:
        out.append("low signal-to-noise ratio (noisy room or very quiet voice): pitch and phone results are less reliable")
    if q["clipped_fraction"] > 0.001:
        out.append("the recording clips (too loud): pronunciation and pitch results are less reliable")
    if q["speech_level_dbfs"] < -40:
        out.append("very quiet recording")
    return out
