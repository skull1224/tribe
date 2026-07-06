# Methodology

## Visual Timeline

Each available video is sampled into 8 evenly spaced frames. For each frame, the pipeline records:

- OCR text using Tesseract via `pytesseract`
- Whether OCR text matches advertiser terms
- CLIP prompt-difference scores for people, face closeups, visual mood, product packshot, and visible text/logo cues
- HSV color features: saturation, brightness, warm-pixel fraction, and cool-pixel fraction

CLIP scores are computed as:

```text
mean(similarity(frame, positive_prompts)) - mean(similarity(frame, negative_prompts))
```

## Audio Timeline

The pipeline reuses existing CLAP segment embeddings from `tmp/clap_audio_baseline/clap_audio_embeddings`.

Each segment is approximately 10 seconds. For each segment, the pipeline computes prompt-difference scores for:

- upbeat audio
- calm audio
- dramatic audio
- humorous audio
- speech/voiceover
- music-forward audio

## Summary Table

`video_timeline_summary.csv` compresses the frame and audio timelines into one row per video.

The current summary thresholds are:

| Feature | Threshold |
|---|---:|
| broad people presence | `clip_people_present_score >= .02` |
| strong people presence | `clip_people_present_score >= .055` |
| face closeup | `clip_face_closeup_score >= 0` |
| high saturation | `saturation_mean >= .45` |

These thresholds are pragmatic cutoffs for exploratory analysis. They should be treated as interpretable controls, not final supervised detectors.
