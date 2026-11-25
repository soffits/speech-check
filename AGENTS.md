# speech-check: notes for maintainers

- `src/speech_check/analyze.py` is the pipeline; each signal is one module and returns a plain dict that the schema
  (`result.schema.json`) describes. Change a field: update the schema, `docs/schema.md`, `report.py` and the tests together.
  Fields are only added within a schema version.
- Offline rule: only `speech-check setup` may reach the network. `cli.main` calls `config.configure_environment` before any
  model library is imported; keep model imports lazy.
- The ASR engine lives only in `asr.py`. Weights are non-commercial: do not publish converted weights or ship model files in the repo.
- Pronunciation thresholds in `pronunciation.py` are uncalibrated choices (see `docs/evaluation.md`); do not tighten claims
  in `reliability.py` without new measurements.
- Tests: `uv run pytest` needs no models. `-m integration` needs set-up models and a Piper voice.
- Dependencies are pinned in `uv.lock`; update with `uv lock --upgrade` and rerun the integration test.
