# AdMeter Timeline Feature Summary

This folder summarizes interpretable, timestamp-level features extracted from the AdMeter video set.

The goal is not to replace TRIBE. These features are intended as readable baseline/control variables around visible brand text, human presence, visual mood, color saturation, and audio mood.

## Outputs

| File | Description |
|---|---|
| `video_timeline_summary.csv` | One row per ad. Includes timestamp ranges for OCR brand text, people/face cues, high-saturation windows, and dominant CLIP/CLAP mood families. |
| `csv_overview.md` | CSV shape, column groups, year-level counts, feature coverage, and mood distributions. |
| `csv_by_year_summary.csv` | One row per year with cue counts and saturation/mood summary. |
| `csv_feature_coverage.csv` | One row per CSV column with non-empty counts and interpretation notes. |
| `examples.md` | Human-readable example rows from the summary table. |
| `data_dictionary.md` | Column definitions and interpretation cautions. |
| `methodology.md` | Extraction pipeline and prompt-score definitions. |

The full frame-level and audio-segment-level files are generated locally by:

```bash
.venv/bin/python scripts/admeter_timeline/ad_timeline_features.py
```

Generated local outputs:

```text
tmp/ad_timeline_features/frame_timeline.csv
tmp/ad_timeline_features/audio_timeline.csv
tmp/ad_timeline_features/video_timeline_summary.csv
```

## Current Coverage

| Metric | Count |
|---|---:|
| Videos processed | 402 |
| Frame rows | 3,216 |
| Audio segment rows | 2,073 |
| Video summary rows | 402 |
| Ads with OCR-readable advertiser text | 89 |
| Ads with strong people-presence windows | 256 |
| Ads with face-closeup windows | 200 |
| Ads with high-saturation windows | 228 |

## How To Read The Summary

`ocr_brand_time_ranges` marks windows where Tesseract OCR read advertiser text. This catches readable text such as "T-Mobile" or "Verizon", but it does not guarantee detection of graphic-only logo marks.

`people_strong_ranges_score_gt_0_055` marks windows where the CLIP people-presence score is high relative to this dataset. `face_closeup_ranges_score_gt_0_00` is stricter in a different way: it captures frames that are closer to face-closeup prompts than non-face prompts.

`visual_top_score_family` and `audio_top_score_family` are coarse labels from prompt similarity, not supervised human annotations. Use them as interpretable covariates or diagnostic baselines.

## Recommended Use In The AdMeter Analysis

Use this table as a control layer:

1. Fit baseline models using interpretable timing features only.
2. Add TRIBE features.
3. Report whether TRIBE improves over visible brand text, human presence, color intensity, and audiovisual mood controls.

This supports the claim that TRIBE is not merely learning simple production cues such as "people appear" or "brand text appears late".
