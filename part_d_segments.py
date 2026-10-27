"""Part D (tasks 1-4) - cluster the 24 hubs into operating segments and check the tier labels.

Run from the repository root (after part_a_clean_eda.py):  python part_d_segments.py
Writes outputs/hub_profile.csv, part_d_separation.csv, hub_segments.csv,
segment_profile.csv and tier_vs_segment_mismatch.csv
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEED = 42
K_SEGMENTS = 3  # fixed by the business: operations runs exactly three playbooks
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]  # lowest -> highest mean delay rate
TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def fit_kmeans(matrix, k):
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(matrix)


def across_over_within(matrix, labels):
    """Mean distance over hub pairs in different clusters / mean over pairs in the same cluster."""
    across, within = [], []
    for i, j in combinations(range(len(matrix)), 2):
        dist = np.linalg.norm(matrix[i] - matrix[j])
        (within if labels[i] == labels[j] else across).append(dist)
    return np.mean(across) / np.mean(within)


def main():
    os.makedirs("outputs", exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # 1. Hub profile, one row per hub in ascending hub_id order.
    profile = table.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index().sort_values("hub_id").reset_index(drop=True)
    profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

    # 2. Scale the four profile columns and cluster into K = 3 segments.
    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    labels = fit_kmeans(scaled, K_SEGMENTS)

    # 3. Separation evidence for K = 2..5 on the same scaled matrix.
    separation_rows = []
    for k in [2, 3, 4, 5]:
        k_labels = fit_kmeans(scaled, k)
        sizes = sorted(np.bincount(k_labels), reverse=True)
        separation_rows.append({
            "K": k,
            "across_over_within": across_over_within(scaled, k_labels),
            "cluster_sizes": "-".join(str(s) for s in sizes),
        })
    pd.DataFrame(separation_rows).round(4).to_csv("outputs/part_d_separation.csv", index=False)

    # 4. Name clusters by their mean hub delay rate, then compare with the tier labels.
    profile["cluster"] = labels
    order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    names = {cluster: SEGMENT_NAMES[rank] for rank, cluster in enumerate(order)}
    profile["segment"] = profile["cluster"].map(names)

    segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
    segments.to_csv("outputs/hub_segments.csv", index=False)

    segment_profile = profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    ).reindex(SEGMENT_NAMES).reset_index()
    segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

    mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_TO_SEGMENT)]
    mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print(f"segments {segment_profile.set_index('segment')['hubs'].to_dict()}; "
          f"mismatches {mismatch['hub_id'].tolist()}")


if __name__ == "__main__":
    main()
