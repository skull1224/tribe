# Local Run Guide

This repository can be installed locally on macOS, but the official docs leave out a few constraints that matter in practice:

- `tribev2` itself expects Python 3.11+.
- The pretrained checkpoint is hosted on Hugging Face as `facebook/tribev2`.
- Audio/video transcription depends on `whisperx`, but `whisperx` upgrades PyTorch beyond the version pinned by `tribev2`.
- The text encoder is `meta-llama/Llama-3.2-3B`, which requires a Hugging Face account with access granted.

## 1. Base environment

```bash
cd /Users/seong-yeob/PycharmProjects/tribe
./scripts/setup_env.sh
```

This creates `./.venv` and installs `tribev2` in editable mode.

## 2. Optional WhisperX environment

If you want text extraction from audio or video, create a separate environment:

```bash
cd /Users/seong-yeob/PycharmProjects/tribe
./scripts/setup_whisperx.sh
```

The code will automatically use `./.whisperx-venv/bin/python -m whisperx` when that environment exists.

## 3. Minimal runnable path

The smallest practical inference path is audio or video with `--skip-text`, because it avoids both WhisperX and the gated Llama text encoder:

```bash
.venv/bin/python scripts/run_inference.py \
  --audio /absolute/path/to/sample.wav \
  --skip-text \
  --device auto \
  --extractor-device auto
```

For video:

```bash
.venv/bin/python scripts/run_inference.py \
  --video /absolute/path/to/sample.mp4 \
  --skip-text \
  --device auto \
  --extractor-device auto
```

## 4. Full multimodal path

If you want the full text/audio/video pipeline:

1. Run `./scripts/setup_whisperx.sh`.
2. Authenticate with Hugging Face for gated Meta Llama access.
3. Run the inference script without `--skip-text`.

Example:

```bash
.venv/bin/python scripts/run_inference.py \
  --video /absolute/path/to/sample.mp4 \
  --device auto \
  --extractor-device auto
```

## 5. Download size notes

Expect large first-run downloads:

- `facebook/tribev2`: about 0.7 GB
- `facebook/w2v-bert-2.0`: about 4.7 GB
- `facebook/vjepa2-vitg-fpc64-256`: about 20.6 GB
- `meta-llama/Llama-3.2-3B`: about 12.9 GB and gated access

Predictions are saved to `./outputs/tribev2_predictions.npz`, and segment metadata is saved next to it as CSV.
