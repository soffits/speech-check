"""Command line: analyze, setup, doctor, schema."""
from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from . import __version__, config


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    # Only `setup` may touch the network; every other command is forced offline before any model library loads.
    config.configure_environment(offline=args.command != "setup")
    return {"analyze": _analyze, "setup": _setup, "doctor": _doctor, "schema": _schema}[args.command](args)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speech-check", description="Analyse a speaking-practice recording locally: "
                                "verbatim transcript, fluency, phone-level pronunciation hints and intonation, as JSON.")
    p.add_argument("--version", action="version", version=f"speech-check {__version__}")
    sub = p.add_subparsers(dest="command")

    a = sub.add_parser("analyze", help="analyse one audio file (any format ffmpeg reads)")
    a.add_argument("audio")
    a.add_argument("--lang", choices=config.LANGUAGES, default="en", help="spoken language (fr: pronunciation part is uncalibrated)")
    ref = a.add_mutually_exclusive_group()
    ref.add_argument("--reference", metavar="TEXT", help="read-aloud mode: the text the speaker was meant to read")
    ref.add_argument("--reference-file", metavar="PATH", help="read-aloud mode: file holding the reference text")
    a.add_argument("--format", choices=("text", "json"), default="text", help="what to print on stdout (default: text summary)")
    a.add_argument("--json-out", metavar="PATH", help="also write the full JSON result here")
    a.add_argument("--asr-model", choices=sorted(config.ASR_MODELS), default=config.DEFAULT_ASR_MODEL)
    a.add_argument("--no-pronunciation", action="store_true", help="skip the phone-level part (fastest)")
    a.add_argument("--no-intonation", action="store_true")
    a.add_argument("--threads", type=int, help="CPU threads for the phone model (default: all)")
    a.add_argument("--quiet", action="store_true", help="no progress lines on stderr")

    s = sub.add_parser("setup", help="download and prepare the models (the only command that uses the network)")
    s.add_argument("--asr-model", choices=sorted(config.ASR_MODELS), default=config.DEFAULT_ASR_MODEL)

    sub.add_parser("doctor", help="check that everything needed to run offline is in place")
    sub.add_parser("schema", help="print the JSON Schema of the result")
    return p


def _analyze(args) -> int:
    from . import analyze
    from .audio import AudioError

    reference = args.reference
    if args.reference_file:
        reference = Path(args.reference_file).read_text(encoding="utf-8").strip()
    log = (lambda msg: None) if args.quiet else (lambda msg: print(f"[speech-check] {msg}", file=sys.stderr, flush=True))
    try:
        result = analyze.run(args.audio, lang=args.lang, reference=reference or None,
                             asr_model=args.asr_model, pronunciation_on=not args.no_pronunciation,
                             intonation_on=not args.no_intonation, threads=args.threads, log=log)
    except (AudioError, RuntimeError, ValueError) as exc:
        print(f"speech-check: {exc}", file=sys.stderr)
        return 1
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        print(result["summary_text"])
        print(f"\n(total {result['timing_s']['total']} s)")
    return 0


def _setup(args) -> int:
    from . import asr, phone_model

    config.home().mkdir(parents=True, exist_ok=True)
    print(f"model cache: {config.home()}")
    print(f"downloading phone model {config.PHONE_MODEL} ...")
    phone_model.download()
    print(f"preparing CrisperWhisper {args.asr_model} (download + one-time CTranslate2 conversion) ...")
    import numpy as np
    asr.transcribe(np.zeros(16000, dtype=np.float32), "en", model=args.asr_model)
    print("done. The CrisperWhisper weights are under a non-commercial research licence: see docs/licences.md.")
    return _doctor(args)


def _doctor(args) -> int:
    import shutil

    ok = True

    def check(name: str, fn):
        nonlocal ok
        try:
            detail = fn()
            print(f"ok    {name}" + (f": {detail}" if detail else ""))
        except Exception as exc:  # report every failing prerequisite, not just the first
            ok = False
            print(f"FAIL  {name}: {exc}")

    def need(binary: str):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} not on PATH")
        return shutil.which(binary)

    def espeak():
        from .phonetics import phones_for_words
        return " ".join(phones_for_words(["think"], "en")[0])

    def phone_cached():
        from huggingface_hub import snapshot_download
        snapshot_download(config.PHONE_MODEL, allow_patterns=["vocab.json"], local_files_only=True)
        if not list((config.home() / "hf" / "hub").glob("models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/*/pytorch_model.bin")) \
                and not list((config.home() / "hf" / "hub").glob("models--facebook--wav2vec2-lv-60-espeak-cv-ft/snapshots/*/*.safetensors")):
            raise RuntimeError("weights missing; run `speech-check setup`")

    def asr_cached():
        size = getattr(args, "asr_model", config.DEFAULT_ASR_MODEL)
        found = list((config.home() / "crisperwhisper").glob(f"nyralabs--CrisperWhisper2.0_{size}_int8_*/.conversion_complete"))
        if not found:
            raise RuntimeError(f"CrisperWhisper {size} not prepared; run `speech-check setup --asr-model {size}`")
        return size

    check("ffmpeg", lambda: need("ffmpeg"))
    check("espeak-ng (bundled library)", espeak)
    check("parselmouth", lambda: __import__("parselmouth").__version__)
    check("ASR model", asr_cached)
    check("phone model", phone_cached)
    print(f"model cache: {config.home()}")
    return 0 if ok else 1


def _schema(args) -> int:
    print(resources.files("speech_check").joinpath("result.schema.json").read_text(encoding="utf-8"))
    return 0
