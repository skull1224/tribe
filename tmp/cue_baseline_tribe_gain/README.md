# Cue baseline + TRIBE gain

- Sample: n=398 for continuous ridge; n=210 for within-year top/bottom 25%.
- Baseline: one production cue at a time.
- Added feature block: 35 TRIBE 5ROI absolute summary features.
- CV: 10-fold shuffled CV and LOO CV; no year fixed effects.
- PCA: K=50 rule retained, but cue+TRIBE 5ROI has 36 features, so PCA is not applied.

## Output figures
- cue_plus_tribe_delta_heatmap_10fold.png
- cue_plus_tribe_topbottom25_auc_slope_10fold.png
- cue_plus_tribe_delta_auc_bar_10fold.png
- cue_plus_tribe_delta_heatmap_loo.png
- cue_plus_tribe_topbottom25_auc_slope_loo.png
- cue_plus_tribe_delta_auc_bar_loo.png

## Largest 10-fold direct top/bottom25 AUC gains
```
             cue_label  baseline_auc  plus_tribe_auc  delta_auc  delta_accuracy
         Face close-up         0.431           0.745      0.314           0.229
      Non-funny visual         0.438           0.748      0.310           0.262
         Strong people         0.438           0.742      0.304           0.219
      Non-upbeat audio         0.461           0.758      0.296           0.214
High saturation window         0.466           0.748      0.282           0.148
        OCR brand text         0.478           0.739      0.260           0.129
       Mean saturation         0.551           0.743      0.191           0.148
```

## Interpretation note
These are incremental predictive gains, not causal effects. They answer whether TRIBE adds out-of-fold predictive signal beyond each simple cue alone.
