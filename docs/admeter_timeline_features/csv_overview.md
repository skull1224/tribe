# CSV Overview

`video_timeline_summary.csv` is the compact GitHub-shareable result table for the AdMeter timeline feature extraction.

## Shape And Grain

- Rows: 402 ads
- Columns: 19
- Grain: one row per ad, identified by `year`, `rank`, and `source_key`.
- Timestamp ranges use seconds in the format `start-end seconds`; multiple windows are separated by semicolons.

## Column Groups

- Metadata: `year`, `rank`, `advertiser`, `title`, `source_key`.
- OCR/logo proxy: `ocr_brand_time_ranges`.
- People/face timing: `people_*`, `face_closeup_*`.
- Color timing: `saturation_*`, `high_saturation_*`.
- Coarse prompt labels: `visual_top_score_family`, `audio_top_score_family`.

## Year-Level Summary

| year | n_ads | ocr_brand_ads | strong_people_ads | face_closeup_ads | high_saturation_ads | mean_saturation | median_saturation | dominant_visual_mood | dominant_audio_mood |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 62 | 12 | 40 | 30 | 38 | 0.3064 | 0.2957 | funny_playful | upbeat_audio |
| 2021 | 56 | 13 | 36 | 27 | 26 | 0.3061 | 0.2954 | funny_playful | upbeat_audio |
| 2022 | 64 | 14 | 40 | 34 | 34 | 0.2904 | 0.2911 | funny_playful | upbeat_audio |
| 2023 | 51 | 9 | 32 | 20 | 25 | 0.312 | 0.298 | funny_playful | upbeat_audio |
| 2024 | 59 | 17 | 34 | 28 | 40 | 0.3388 | 0.3076 | funny_playful | upbeat_audio |
| 2025 | 57 | 14 | 40 | 29 | 33 | 0.341 | 0.3308 | funny_playful | upbeat_audio |
| 2026 | 53 | 10 | 34 | 32 | 32 | 0.326 | 0.326 | funny_playful | upbeat_audio |

## Feature Coverage

| column | group | non_empty_ads | pct_of_ads |
| --- | --- | --- | --- |
| advertiser | metadata | 402 | 1.0 |
| audio_top_score_family | coarse_label | 402 | 1.0 |
| face_closeup_peak_score | numeric_score | 402 | 1.0 |
| face_closeup_peak_time | peak_timestamp | 402 | 1.0 |
| face_closeup_ranges_score_gt_0_00 | timestamp_range | 200 | 0.498 |
| high_saturation_ranges_gt_0_45 | timestamp_range | 228 | 0.567 |
| ocr_brand_time_ranges | timestamp_range | 89 | 0.221 |
| people_peak_score | numeric_score | 402 | 1.0 |
| people_peak_time | peak_timestamp | 402 | 1.0 |
| people_strong_ranges_score_gt_0_055 | timestamp_range | 256 | 0.637 |
| people_time_ranges_score_gt_0_02 | timestamp_range | 398 | 0.99 |
| rank | metadata | 402 | 1.0 |
| saturation_mean | numeric_score | 402 | 1.0 |
| saturation_peak_score | numeric_score | 402 | 1.0 |
| saturation_peak_time | peak_timestamp | 402 | 1.0 |
| source_key | metadata | 402 | 1.0 |
| title | metadata | 402 | 1.0 |
| visual_top_score_family | coarse_label | 402 | 1.0 |
| year | metadata | 402 | 1.0 |

## Visual Mood Distribution

| visual_mood | n_ads |
| --- | --- |
| funny_playful | 366 |
| dramatic_dark | 20 |
| energetic_action | 15 |
| sentimental_warm | 1 |

## Audio Mood Distribution

| audio_mood | n_ads |
| --- | --- |
| upbeat_audio | 365 |
| dramatic_audio | 26 |
| calm_audio | 8 |
| humorous_audio | 3 |

## Companion CSV Files

- `csv_by_year_summary.csv`: one row per year with cue counts and dominant mood labels.
- `csv_feature_coverage.csv`: one row per CSV column with non-empty counts and column group labels.

## Reading Cautions

- Empty timestamp cells mean the cue was not detected under the current threshold, not necessarily absent in the original ad.
- OCR catches readable brand text and can miss graphic-only logos.
- CLIP/CLAP fields are prompt-similarity features, not manually labeled annotations.
