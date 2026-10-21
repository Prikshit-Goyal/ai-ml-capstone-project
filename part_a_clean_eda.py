"""Part A - clean the raw export, build features, and write the first delay evidence.

Run from the repository root:  python part_a_clean_eda.py
Writes data/modelling_table.csv and outputs/part_a_{checks,delay_by_group,correlations}.csv
"""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

DISTANCE_BINS = [0, 200, 400, 600, 2000]
DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]


def r4(value):
    """Round floats to 4 places; leave integers as integers."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), 4)


def main():
    os.makedirs("outputs", exist_ok=True)
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # "Before" measurements: after removing duplicates, before any filling.
    dedup = raw.drop_duplicates()
    dedup_weight = dedup["parcel_weight_kg"]
    stripped_titled = dedup["service_type"].str.strip().str.title()

    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
    table.to_csv("data/modelling_table.csv", index=False)

    # Spread of delivery_hours: 1.5 x IQR rule on unrounded quartiles.
    q1 = table["delivery_hours"].quantile(0.25)
    q3 = table["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((table["delivery_hours"] < lower) | (table["delivery_hours"] > upper)).sum()

    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", len(raw) - len(dedup)),
        ("rows_after_dedup", len(dedup)),
        ("missing_distance_km_before", dedup["distance_km"].isna().sum()),
        ("missing_parcel_weight_before", dedup_weight.isna().sum()),
        ("sentinel_weight_rows", (dedup_weight > 100).sum()),
        ("missing_declared_value_before", dedup["declared_value"].isna().sum()),
        ("missing_payment_mode_before", dedup["payment_mode"].isna().sum()),
        ("service_type_spellings_before", dedup["service_type"].nunique()),
        ("service_type_spellings_after", stripped_titled.nunique()),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(table)),
        ("nulls_in_features_and_target", table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        ("delayed_count", table["is_delayed"].sum()),
        ("delay_rate", table["is_delayed"].mean()),
        ("declared_value_mean_inr", table["declared_value_inr"].mean()),
        ("declared_value_median_inr", table["declared_value_inr"].median()),
        ("log_declared_value_mean", table["log_declared_value"].mean()),
        ("log_declared_value_median", table["log_declared_value"].median()),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", lower),
        ("hours_upper_bound", upper),
        ("hours_outlier_count", outliers),
    ]
    pd.DataFrame(
        {"metric": [m for m, _ in checks], "value": pd.Series([r4(v) for _, v in checks], dtype=object)}
    ).to_csv("outputs/part_a_checks.csv", index=False)

    # Delay rate by service type, hub tier and distance band.
    table["distance_band"] = pd.cut(
        table["distance_km"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=True, include_lowest=True
    )
    groups = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        g = table.groupby(dim, observed=True).agg(
            shipments=("is_delayed", "size"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean"),
        ).reset_index().rename(columns={dim: "group"})
        g.insert(0, "dimension", dim)
        groups.append(g)
    by_group = pd.concat(groups, ignore_index=True)
    by_group["group"] = by_group["group"].astype(str)
    by_group.round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # Correlation matrix of numeric columns (row labels kept as the first column).
    corr = table.drop(columns=["distance_band"]).select_dtypes(include="number").corr().round(4)
    corr.to_csv("outputs/part_a_correlations.csv", index=True, index_label="column")

    print(f"modelling_table: {len(table)} rows, delayed {table['is_delayed'].sum()} "
          f"({table['is_delayed'].mean():.4f})")


if __name__ == "__main__":
    main()
