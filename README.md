# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project — AI & ML Certification Program, E&ICT Academy, IIT Roorkee.

A single pipeline that answers three operations questions on synthetic Delhivery shipment data:
how long a parcel will take (regression), which parcels will miss their promise (classification),
and which hubs need which playbook (clustering).

> The data is synthetic and produced by `generate_data.py`. No real Delhivery records are used.

## How to run

From a fresh clone, at the repository root:

```bash
pip install -r requirements.txt
python generate_data.py        # prints "24 1224 48"; regenerates data/hubs.csv, raw_shipments.csv, new_shipments.csv
python part_a_clean_eda.py     # writes data/modelling_table.csv and outputs/part_a_*.csv
python part_b_regression.py    # writes outputs/part_b_*.csv
python part_c_classification.py  # writes outputs/part_c_*.csv
```

The commands for Part D will be added when that Part is built.

## Data findings

All numbers come from `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv` and
`outputs/part_a_correlations.csv`.

The cleaned modelling table has **1,200** unique shipments, after removing 24 exact duplicates from the 1,224 raw rows.
Of these, **278** are delayed, an overall delay rate of **0.2317**.

1. **Service type.** **Express** is delayed most often: 119 of 243 parcels, a delay rate of **0.4897**.
   Standard follows at 0.2056 and Economy at 0.0857. Express has the tightest promise (36 hours),
   so it has the least slack to absorb distance, weather or sale-period congestion.
2. **Hub tier.** Delay rate rises steeply from **0.0400** at Tier 1, through **0.2150** at Tier 2,
   to **0.4400** at Tier 3. Tier 3 hubs are delayed eleven times as often as Tier 1 hubs.
   Mean delivery time also rises, from 48.5664 hours to 63.4943 hours.
3. **Distance band.** Delay rate climbs with every band: **0.0219** for 0-200 km, **0.1368** for 201-400 km,
   **0.4364** for 401-600 km and **0.8523** for 601+ km. In the 601+ band, 75 of 88 parcels are late.
4. **Strongest correlations with `delivery_hours`.** Ranked by absolute value, excluding the target itself:
   `promised_hours` **0.8372**, `distance_km` **0.4013** and `n_stops` **0.3247**.
   `promised_hours` encodes the service type, which sets the base transit time. Distance and rider-route stops add hours on top of that.
5. **Why `declared_value_inr` is log-transformed.** On the median-filled column, the mean is INR 1,533.05
   and the median is INR 1,121.50. The mean is about 37% above the median, so a few high-value parcels give a long right tail.
   After the natural log, the mean (7.0277) and median (7.0224) almost coincide.
   The log removes the skew so that a handful of expensive parcels cannot dominate a scaled or linear model.

## Model results (regression)

All numbers come from `outputs/part_b_checks.csv`, `part_b_metrics.csv`, `part_b_tree_depth.csv`,
`part_b_gb_learning_rate.csv` and `part_b_cv.csv`. The target is `delivery_hours`; the inputs are the 17 pinned features.

**Split and leakage check.** The data is split into 840 training, 180 validation and 180 test rows.
The `StandardScaler` was fitted on the training rows only:

- The scaled training columns have a largest absolute mean of 0.0.
- The scaled test columns have a largest absolute mean of 0.2187.

The test rows never influenced the scaler.

**Baseline.** Predicting the training mean for every parcel gives a validation R² of **−0.0212** (RMSE 18.8552 hours).
An R² around zero means the model explains none of the variation between parcels.
Any useful model must beat this, and every candidate does by a wide margin.

**Validation comparison (RMSE in hours, R²).**

| Model | Validation RMSE (hours) | Validation R² |
|---|---|---|
| linear regression | 4.8912 | 0.9313 |
| gradient boosting | 4.9165 | 0.9306 |
| random forest | 5.4784 | 0.9138 |
| decision tree (depth 6) | 6.2796 | 0.8867 |

**Linear regression** won on validation with an RMSE of **4.8912 hours**, so it is the final model under the brief's rule.
Scored once on the test rows, it reached R² **0.9493** (RMSE 4.5235 hours, MAE 3.5671 hours).
Linear regression does this well because the delivery time is mostly a sum of separate effects:
base time by service, plus distance, stops, rain, sale period and so on.

**Bias versus variance (decision-tree depth).**

- **Shallow trees are high-bias (depths 1–3).** Training and validation errors are both high and close together.
  At depth 1 the training RMSE is 11.6334 and the validation RMSE 12.1324, a gap of only 0.50.
  The tree is too simple to capture the pattern: it underfits.
- **Depth 6 is the sweet spot**, with the lowest validation RMSE of 6.2796.
  That is 5.85 hours better than depth 1.
- **Deep trees are high-variance (depths 8, 12 and unlimited).** Training error keeps falling while validation error rises again.
  At depth 12 the training RMSE is 1.3613 and the validation RMSE 7.4173.
  With no limit the training RMSE is 0.0 and the validation RMSE 7.7890, a gap of 7.79 hours.
  The unlimited tree memorises the training parcels: it overfits.

**Learning rate for gradient boosting (100 trees).**

| Learning rate | Validation R² |
|---|---|
| 0.01 | 0.7127 |
| 0.1 | **0.9306** |
| 1.0 | 0.8440 |

- **0.1 is the best rate.**
- **0.01 is too slow.** Each tree corrects only 1% of the remaining error, so 100 trees stop far short of a good fit (underfitting).
- **1.0 overshoots.** Each tree applies its full correction, so the model chases noise in the training rows and generalises worse.

**Cross-validation versus validation.** The two best models were compared with 5-fold cross-validation on the 1,020 training-plus-validation rows.
Both models used the same `KFold(shuffle=True, random_state=42)` object.

| Model | Mean R² across folds | Std R² |
|---|---|---|
| gradient boosting | **0.9352** | 0.0044 |
| linear regression | 0.9261 | 0.0048 |

The two rankings disagree. Linear regression is ahead on the single 180-row validation set, but only by 0.0253 hours of RMSE.
Gradient boosting is ahead in cross-validation, by 0.0091 R², and the result is consistent across folds (std below 0.005).

**I would ship gradient boosting**, for three reasons:

- Cross-validation averages over five different 204-row held-out sets drawn from 1,020 rows, so it is a more reliable comparison than one 180-row validation split.
- The validation gap in favour of linear regression is too small to be meaningful.
- Boosted trees can learn interactions that a purely additive linear model cannot. Examples are the extra delay when a sale coincides with Express, or when rain coincides with long distances.

Caveat: the reported test score (0.9493) belongs to linear regression, the model the brief's rule selected.
Gradient boosting would need its own single test evaluation before it goes live.
For transparency, linear regression remains a strong, simple fallback.

## Model results (classification)

All numbers come from `outputs/part_c_models.csv`, `part_c_feature_scores.csv`, `part_c_selectk.csv`,
`part_c_tuning.csv`, `part_c_thresholds.csv`, `part_c_final_test.csv` and `part_c_final_config.csv`.
The target is `is_delayed`. The split is stratified: 840 / 180 / 180 rows, with 195 / 42 / 41 delayed parcels.

**Blind baseline.** Predicting "on time" for every parcel scores **0.7667 accuracy** on validation (TN 138, FP 0, FN 42, TP 0).
Its recall is 0, so it catches none of the 42 delayed parcels, and its accuracy only reflects that about 77% of parcels are on time.
Accuracy alone therefore hides a useless model. F1 and recall are the metrics that matter.

| Model (validation, threshold 0.5) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| blind_all_on_time | 0.7667 | 0.0 | 0.0 | 0.0 |
| logistic_regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| decision_tree_depth_6 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| random_forest_default | 0.9111 | 0.8421 | 0.7619 | 0.8000 |

**Noise columns.** The five bookkeeping columns together take **0.2253** of the default forest's importance:

| Column | Importance |
|---|---|
| `label_print_batch` | 0.0462 |
| `scanner_build_no` | 0.0377 |
| `tape_roll_no` | 0.0478 |
| `sticker_serial_mod` | 0.0496 |
| `shelf_slot_no` | 0.0440 |

The forest spreads importance onto random columns simply because it can split on them.
By F-score, **all five fall outside the top 7**: their ranks are 8, 12, 14, 16 and 17, and every F-score is below 4.6.
The top 7 by F-score are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`.
The smallest F-score among them is 36.7281.

**What the k table shows.** Validation F1 by number of selected columns:

| k | Validation F1 |
|---|---|
| 5 | 0.6988 |
| 7 | **0.8333** |
| 10 | 0.8140 |
| 12 | 0.8049 |
| 17 | 0.8000 |

Five columns is too few, because `rain_flag` and `sale_event_flag` are lost.
Seven columns is the best. Adding the weaker and noise columns back slowly lowers F1.

**Tuning versus the untuned forest.** Both searches ran on the 7 selected columns:

| Search | Candidates | Fits | Best CV accuracy | Validation F1 |
|---|---|---|---|---|
| Grid search | 27 | 135 | **0.8952** | 0.7750 |
| Randomized search | 9 | 45 | 0.8905 | 0.7901 |
| Untuned forest (100 trees, no depth limit, `min_samples_leaf` 1) | 1 | 1 | — | **0.8333** |

Tuning did **not** beat the untuned forest on validation. The final configuration is therefore `default_forest_same_7_features`.
This shows that on a 180-row validation set with only 42 delayed parcels, a few parcels decide the winner.
The searches optimised cross-validated *accuracy* on training folds, which is not the same as F1 on validation.
The gaps between the three candidates are small enough to be noise, not proof that tuning is harmful.

**Alert threshold.** The operations rule is to take the highest threshold whose validation recall is at least 0.80, because missing a delayed parcel costs more than an unnecessary call.

| Threshold | Precision | Recall | Meets the rule? |
|---|---|---|---|
| 0.4 | 0.7400 | 0.8810 | yes |
| **0.5** | **0.7955** | **0.8333** | **yes, the highest that does** |
| 0.6 | 0.8889 | 0.7619 | no, recall drops below 0.80 |

The chosen threshold is **0.5**. Raising it to 0.6 would make alerts more precise but would miss 3 more delayed parcels (FN 10 instead of 7).
Lowering it would catch a few more delayed parcels at the cost of many extra calls. At 0.1 there are 44 false alarms.

**Test result (one evaluation, threshold 0.5).** Precision is **0.6889**, recall **0.7561** and F1 **0.7209** (TN 125, FP 14, FN 10, TP 31).
Test recall is lower than validation recall (0.7561 versus 0.8333), so on new data the model fell short of the 0.80 target.
The test set has only 41 delayed parcels, so each missed parcel moves recall by about 0.024.
The gap is 3 parcels' worth, well within what one small split can produce.
But it is a warning: the 0.80 recall target was met on the split used to choose the threshold, not proven on unseen data.
The model was not re-tuned after seeing the test result.

## Operations recommendation

*To be completed in Part D.*

## Production plan

*To be completed in Part D.*

## Limitations

*To be completed in Part D.*
