#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRIBE v2 inference on one text, audio, or video input."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=Path, help="Path to a .txt file.")
    source.add_argument("--audio", type=Path, help="Path to an audio file.")
    source.add_argument("--video", type=Path, help="Path to a video file.")
    parser.add_argument(
        "--checkpoint",
        default="facebook/tribev2",
        help="Local checkpoint directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best.ckpt",
        help="Checkpoint filename inside the checkpoint directory or repo.",
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path("./cache"),
        help="Directory used for downloaded weights and extracted features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/tribev2_predictions.npz"),
        help="Compressed NumPy output path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Model device. Use auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--extractor-device",
        default="auto",
        help="Feature extractor device. Use auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help=(
            "Skip transcription and text features for audio/video inputs. "
            "This avoids WhisperX and the gated Llama text encoder."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the tqdm prediction progress bar.",
    )
    return parser.parse_args()


def build_segment_rows(segments: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, segment in enumerate(segments):
        rows.append(
            {
                "index": idx,
                "start": getattr(segment, "start", None),
                "duration": getattr(segment, "duration", None),
                "timeline": getattr(segment, "timeline", None),
                "n_events": len(getattr(segment, "ns_events", [])),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    from tribev2 import TribeModel

    args.cache_folder.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = TribeModel.from_pretrained(
        args.checkpoint,
        checkpoint_name=args.checkpoint_name,
        cache_folder=args.cache_folder,
        device=args.device,
        extractor_device=args.extractor_device,
    )

    if args.text is not None:
        events = model.get_events_dataframe(text_path=str(args.text))
    elif args.audio is not None:
        events = model.get_events_dataframe(
            audio_path=str(args.audio), include_text=not args.skip_text
        )
    else:
        events = model.get_events_dataframe(
            video_path=str(args.video), include_text=not args.skip_text
        )

    preds, segments = model.predict(events=events, verbose=not args.quiet)

    np.savez_compressed(args.output, preds=preds)
    segments_path = args.output.with_suffix(".segments.csv")
    pd.DataFrame(build_segment_rows(segments)).to_csv(segments_path, index=False)

    print(f"predictions: {preds.shape}")
    print(f"saved predictions to {args.output}")
    print(f"saved segment summary to {segments_path}")


if __name__ == "__main__":
    main()
