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
```

The commands for Parts B–D will be added as those Parts are built.

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

*To be completed in Part B.*

## Model results (classification)

*To be completed in Part C.*

## Operations recommendation

*To be completed in Part D.*

## Production plan

*To be completed in Part D.*

## Limitations

*To be completed in Part D.*
