"""Where models live and how the process is kept offline.

Everything that downloads (model files) happens only in `speech-check setup`; every other
command runs with the Hugging Face hub forced offline so a recording can never leave the machine.
"""
from __future__ import annotations

import os
from pathlib import Path

ASR_MODELS = {
    "small": "nyralabs/CrisperWhisper2.0_small",
    "medium": "nyralabs/CrisperWhisper2.0_medium",
    "turbo": "nyralabs/CrisperWhisper2.0_turbo",
    "large": "nyralabs/CrisperWhisper2.0_large",
}
DEFAULT_ASR_MODEL = "medium"
PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
LANGUAGES = ("en", "fr")


def home() -> Path:
    """Model cache root: $SPEECH_CHECK_HOME, else ~/.cache/speech-check."""
    return Path(os.environ.get("SPEECH_CHECK_HOME", "~/.cache/speech-check")).expanduser()


def configure_environment(*, offline: bool) -> None:
    """Point the model libraries at our cache; with `offline`, forbid any network access.

    Must run before huggingface_hub or transformers are imported (they read these at import).
    """
    root = home()
    os.environ["HF_HOME"] = str(root / "hf")
    os.environ["CRISPERWHISPER_CACHE"] = str(root / "crisperwhisper")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
