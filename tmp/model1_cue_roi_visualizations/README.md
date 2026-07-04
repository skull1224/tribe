# Model 1 cue vs TRIBE 5ROI activation

- Source rows: 404; complete analysis rows: 396.
- ROI activation degree: mean absolute value across the Model 2 TRIBE 5ROI summary features for each ROI.
- Cue present/high effects use present minus absent; non-upbeat and non-funny use those categories minus their references.
- saturation_mean is continuous in Model 1; the bar plot uses a median split only for visualization.

## Files

- model1_cue_5roi_abs_activation_bars.png
- model1_cue_5roi_cohens_d_heatmap.png
- model1_cue_5roi_cohens_d_heatmap_clean.png
- model1_saturation_mean_5roi_correlation.png
- model1_saturation_mean_5roi_correlation_report.png
- model1_cue_5roi_abs_activation_summary.csv
- model1_cue_5roi_abs_activation_stats.csv

## Largest absolute cue effects

             cue              roi  n_present  n_absent  cohens_d_present_minus_absent  mean_diff_present_minus_absent  p_value
Non-upbeat audio            dlPFC         36       360                        -0.4752                         -0.0141   0.0034
Non-funny visual         Amygdala         35       361                         0.4720                          0.0096   0.0179
Non-funny visual      Hippocampus         35       361                        -0.3831                         -0.0055   0.0088
Non-upbeat audio         Amygdala         36       360                         0.3829                          0.0078   0.0515
  OCR brand text         Amygdala         88       308                         0.3669                          0.0075   0.0077
  OCR brand text Ventral striatum         88       308                        -0.3006                         -0.0048   0.0114
Non-funny visual            dlPFC         35       361                        -0.2601                         -0.0078   0.1397
  OCR brand text            dlPFC         88       308                        -0.2583                         -0.0077   0.0224
   Face close-up         Amygdala        196       200                         0.2474                          0.0051   0.0146
   Face close-up      Hippocampus        196       200                         0.2343                          0.0033   0.0203
Non-funny visual Ventral striatum         35       361                        -0.2215                         -0.0035   0.1700
Non-upbeat audio Ventral striatum         36       360                        -0.2124                         -0.0034   0.2006
