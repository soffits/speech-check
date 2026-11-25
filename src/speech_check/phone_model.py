"""The wav2vec2 phone recogniser: audio in, frame-wise phone log-posteriors out."""
from __future__ import annotations

import json

import numpy as np

from . import config
from .audio import SAMPLE_RATE

FRAME_SECONDS = 0.02


class PhoneModel:
    def __init__(self, threads: int | None = None):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(config.PHONE_MODEL).eval()
            with open(hf_hub_download(config.PHONE_MODEL, "vocab.json")) as fh:
                vocab = json.load(fh)
        except OSError as exc:
            raise RuntimeError(f"phone model not available offline; run `speech-check setup` ({exc})") from exc
        self.vocab = tuple(tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]))
        self.blank = vocab["<pad>"]

    def log_probs(self, samples: np.ndarray) -> np.ndarray:
        """(frames, vocab) log-posteriors, 20 ms per frame."""
        torch = self._torch
        x = np.asarray(samples, dtype=np.float32)
        min_len = int(0.4 * SAMPLE_RATE)
        if len(x) < min_len:
            x = np.pad(x, (0, min_len - len(x)))
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(x)[None]).logits[0].float()
            return torch.log_softmax(logits, dim=-1).numpy()


def download() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(config.PHONE_MODEL, allow_patterns=["*.json", "*.bin", "*.txt", "*.safetensors"])
