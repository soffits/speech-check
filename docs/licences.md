# Licences

| Component | Licence | Notes |
|---|---|---|
| CrisperWhisper 2.0 inference code (`crisperwhisper`) | MIT | |
| CrisperWhisper 2.0 **model weights and their outputs** | nyra health Non-Commercial Research License v1.0 (17 Jul 2026) | Free for non-commercial use, including private practice. No commercial gain, no hosted service or API for third parties, no use of outputs to train commercial models. Derivatives (quantised or converted weights, anything trained on outputs) stay under the same licence and may not be released under MIT/Apache/BSD. Redistribution requires the full licence text and a NOTICE line (see `NOTICE.md`). Commercial licences: licensing@nyra-labs.com |
| wav2vec2-lv-60-espeak-cv-ft | Apache-2.0 | facebook, via Hugging Face |
| espeak-ng (wheel `espeakng-loader`) | GPL-3 | bundled library, used through phonemizer-fork |
| phonemizer-fork | GPL-3 | |
| praat-parselmouth | GPL-3 or later | |
| torch, transformers, numpy, rapidfuzz | BSD / Apache-2.0 / BSD / MIT | |
| ctranslate2, huggingface_hub, jsonschema (dev) and other transitive dependencies | MIT / Apache-2.0 / BSD | see each package's metadata |
| OpenPronounce (adapted logic) | MIT | see `NOTICE.md` |
| Dev only: piper-tts | GPL-3 | synthetic test clips; voices are separate downloads |

The code of this project is AGPL-3.0-or-later. It is compatible with the GPL-3 components above (GPLv3 section 13 allows combining GPL-3 and AGPL-3 code). The models are downloaded by `speech-check setup`, never shipped here, so their terms apply to you separately: with the default engine the tool is for non-commercial use. The ASR engine is isolated in `src/speech_check/asr.py`;
swapping it for a permissively licensed engine (for example faster-whisper, which drops most fillers and
repetitions) removes the non-commercial restriction on the transcript but not the GPL components.
