# Data Dictionary

`video_timeline_summary.csv` contains one row per ad.

| Column | Meaning |
|---|---|
| `year` | AdMeter year. |
| `rank` | AdMeter rank within year. |
| `advertiser` | Advertiser/brand label from the AdMeter metadata. |
| `title` | Ad title. |
| `source_key` | Stable source identifier used in the local pipeline. |
| `ocr_brand_time_ranges` | Time windows where OCR text matched advertiser terms. Empty means no readable advertiser text was detected. |
| `people_peak_time` | Center timestamp of the highest CLIP people-presence score. |
| `people_peak_score` | Highest CLIP people-presence score. |
| `people_time_ranges_score_gt_0_02` | Broad people-presence windows. This threshold is intentionally sensitive. |
| `people_strong_ranges_score_gt_0_055` | Stronger people-presence windows, roughly the upper tail of frame scores. |
| `face_closeup_peak_time` | Center timestamp of the highest CLIP face-closeup score. |
| `face_closeup_peak_score` | Highest CLIP face-closeup score. |
| `face_closeup_ranges_score_gt_0_00` | Windows where face-closeup prompts exceed non-face prompts. |
| `saturation_mean` | Mean HSV saturation over the sampled frames. |
| `saturation_peak_time` | Center timestamp of the most saturated sampled frame. |
| `saturation_peak_score` | Highest sampled-frame saturation. |
| `high_saturation_ranges_gt_0_45` | Windows where mean frame saturation is at least `.45`. |
| `visual_top_score_family` | Highest average CLIP visual mood family among funny/playful, sentimental/warm, energetic/action, and dramatic/dark. |
| `audio_top_score_family` | Highest average CLAP audio mood family among upbeat, calm, dramatic, and humorous. |

## Important Cautions

- OCR detects readable text, not all logos.
- CLIP and CLAP scores are prompt-similarity scores, not manually labeled ground truth.
- Timestamp windows are approximate because each video is sampled at 8 evenly spaced frames.
- Audio timestamps use existing CLAP 10-second segment embeddings.
