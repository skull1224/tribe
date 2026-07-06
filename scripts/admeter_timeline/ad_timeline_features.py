from __future__ import annotations

from ad_timeline_audio_summary import audio_rows, summary_rows
from ad_timeline_common import (
    CLAP_SEGMENT_SEC,
    FRAME_COUNT,
    OUT_DIR,
    encode_clap_text,
    encode_clip_text,
    read_ads,
    write_csv,
)
from ad_timeline_visual import frame_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ads = read_ads()
    clip_vectors = encode_clip_text()
    clap_vectors = encode_clap_text()
    frames = []
    audio = []
    for index, ad in enumerate(ads, start=1):
        frames.extend(frame_rows(ad, clip_vectors))
        audio.extend(audio_rows(ad, clap_vectors))
        if index % 20 == 0 or index == len(ads):
            print(f"timeline progress {index}/{len(ads)}", flush=True)
    summary = summary_rows(frames, audio)
    write_csv(OUT_DIR / "frame_timeline.csv", frames)
    write_csv(OUT_DIR / "audio_timeline.csv", audio)
    write_csv(OUT_DIR / "video_timeline_summary.csv", summary)
    (OUT_DIR / "summary.md").write_text(
        "\n".join(
            [
                "# Ad timeline features",
                "",
                f"- Videos processed: {len(ads)}",
                (
                    f"- Frame rows: {len(frames)} using {FRAME_COUNT} evenly "
                    "sampled frames per video."
                ),
                (
                    f"- Audio rows: {len(audio)} using existing CLAP "
                    f"{CLAP_SEGMENT_SEC:g}s segments."
                ),
                (
                    "- OCR brand hit means advertiser text was read by "
                    "Tesseract in that frame."
                ),
                (
                    "- CLIP/CLAP scores are positive-prompt similarity "
                    "minus negative-prompt similarity."
                ),
            ]
        )
        + "\n"
    )
    print(f"wrote {OUT_DIR}")
    print(
        f"frame rows={len(frames)} audio rows={len(audio)} "
        f"summary rows={len(summary)}"
    )


if __name__ == "__main__":
    main()
