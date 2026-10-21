"""Shared constants and preprocessing for the Delay Intelligence pipeline.

Used by every Part: training scripts read the table built with these functions,
and predict_delay.py re-uses them on new parcels so preprocessing is identical.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # network accepts parcels up to 40 kg; anything above 100 is a scanner fault


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians).

    If medians is None they are computed from this table (training history);
    otherwise the passed dictionary is used (scoring new parcels).
    """
    df = raw.copy()

    # 1. exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. one spelling per service type
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. missing payment mode -> explicit placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. "₹1,290" -> 1290.0
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df = df.drop(columns=["declared_value"])

    # 5. timestamp text -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")

    # 6. sentinel weights are missing values
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan

    # 7. median fill
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub attributes and derived model inputs. Never uses delivery_hours."""
    df = clean.merge(hubs, on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
