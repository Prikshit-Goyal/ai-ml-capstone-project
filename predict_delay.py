"""Part D (task 6) - score new parcels and turn them into an operations action list.

Run from the repository root (after package_model.py and part_d_segments.py):  python predict_delay.py
Writes outputs/new_shipment_actions.csv
"""
import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments


def choose_action(flagged, segment):
    if not flagged:
        return "No action"
    if segment == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"


def main():
    new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute["column"], impute["median"]))
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

    # Same preprocessing as training, with the stored historical medians.
    clean, _ = clean_shipments(new, medians=medians)
    features = build_features(clean, hubs)

    scored = features[["shipment_id", "hub_id"]].copy()
    scored["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
    scored["flagged"] = (scored["delay_probability"] >= threshold).astype(int)
    scored = scored.merge(segments, on="hub_id", how="left")
    scored["action"] = [choose_action(f, s) for f, s in zip(scored["flagged"], scored["segment"])]

    actions = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    actions.to_csv("outputs/new_shipment_actions.csv", index=False)
    print(f"scored {len(actions)} parcels; flagged {actions['flagged'].sum()}; "
          f"escalations {(actions['action'] == 'Escalate to hub manager').sum()}")


if __name__ == "__main__":
    main()
