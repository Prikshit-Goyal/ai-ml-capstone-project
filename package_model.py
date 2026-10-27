"""Part D (task 5) - refit Part C's final classifier and save it as a reusable artefact.

Run from the repository root (after part_c_classification.py):  python package_model.py
Writes models/delay_model.pkl, delay_threshold.csv, feature_order.csv, impute_values.csv
and outputs/part_d_reload_check.csv
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

SEED = 42
MODEL_PATH = "models/delay_model.pkl"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    threshold = float(config["threshold"])

    # Same stratified split as Part C.
    table = pd.read_csv("data/modelling_table.csv")
    X = table[FEATURES]
    y = table["is_delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    _, X_test, _, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    model = Pipeline([
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(
            random_state=SEED,
            n_estimators=int(config["n_estimators"]),
            max_depth=max_depth,
            min_samples_leaf=int(config["min_samples_leaf"]),
        )),
    ]).fit(X_train, y_train)

    # Save the model and everything predict_delay.py needs to reproduce training-time preprocessing.
    joblib.dump(model, MODEL_PATH)
    pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    _, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
    pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}).to_csv(
        "models/impute_values.csv", index=False
    )

    # Reload and confirm the pickle predicts exactly like the in-memory pipeline.
    reloaded = joblib.load(MODEL_PATH)
    diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
    pd.DataFrame({
        "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
        "value": pd.Series([len(X_test), round(float(diff), 4), threshold], dtype=object),
    }).to_csv("outputs/part_d_reload_check.csv", index=False)

    print(f"saved {MODEL_PATH}; test rows {len(X_test)}; reload diff {diff}; threshold {threshold}")


if __name__ == "__main__":
    main()
