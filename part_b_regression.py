"""Part B - predict delivery_hours and decide which regression model to trust.

Run from the repository root (after part_a_clean_eda.py):  python part_b_regression.py
Writes outputs/part_b_{checks,tree_depth,metrics,gb_learning_rate,cv}.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

TARGET = "delivery_hours"
SEED = 42
TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    os.makedirs("outputs", exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    X = table[FEATURES]
    y = table[TARGET]

    # 1. Split 70 / 15 / 15: train vs temp, then temp -> validation vs test.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

    # 2. Scaler fitted on training rows only; test rows only transformed.
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    checks = pd.DataFrame({
        "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
        "value": pd.Series([
            len(X_train), len(X_val), len(X_test),
            round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
            round(float(np.abs(test_scaled.mean(axis=0)).max()), 4),
        ], dtype=object),
    })
    checks.to_csv("outputs/part_b_checks.csv", index=False)

    # 3. Baseline: predict the training mean for every validation row.
    metric_rows = []
    baseline_pred = np.full(len(y_val), y_train.mean())
    metric_rows.append({"model": "mean_baseline", "split": "val", **regression_metrics(y_val, baseline_pred)})

    # 4. Tree-depth diagnostic: training vs validation RMSE per depth.
    depth_rows = []
    for d in TREE_DEPTHS:
        tree = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_train, y_train)
        depth_rows.append({
            "max_depth": "unlimited" if d is None else d,
            "train_rmse": rmse(y_train, tree.predict(X_train)),
            "val_rmse": rmse(y_val, tree.predict(X_val)),
        })
    depth_table = pd.DataFrame(depth_rows)
    depth_table.round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)
    # Lowest val_rmse; idxmin returns the first (smaller depth) on ties since depths are ascending.
    best_depth = TREE_DEPTHS[int(depth_table["val_rmse"].idxmin())]

    # 5. Four candidates, fitted on train, scored on validation.
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=SEED),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=SEED),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED),
    }
    val_rmse = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        m = regression_metrics(y_val, model.predict(X_val))
        val_rmse[name] = m["RMSE"]
        metric_rows.append({"model": name, "split": "val", **m})

    # 6. Learning-rate experiment for gradient boosting.
    lr_rows = []
    for lr in LEARNING_RATES:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=SEED).fit(X_train, y_train)
        pred = gb.predict(X_val)
        lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
    pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

    # 7. Cross-validate the two best on train+validation with one shared KFold.
    ranking = sorted(val_rmse, key=val_rmse.get)
    X_trval = pd.concat([X_train, X_val])
    y_trval = pd.concat([y_train, y_val])
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name in ranking[:2]:
        scores = cross_val_score(candidates[name], X_trval, y_trval, cv=kfold)
        cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
    pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

    # Final model = lowest validation RMSE, already fitted on train; one test evaluation.
    final_name = ranking[0]
    final_test = regression_metrics(y_test, candidates[final_name].predict(X_test))
    metric_rows.append({"model": final_name, "split": "test", **final_test})
    pd.DataFrame(metric_rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)

    print(f"split {len(X_train)}/{len(X_val)}/{len(X_test)}; best tree depth {best_depth}; "
          f"validation ranking {ranking}; final {final_name} test R2 {final_test['R2']:.4f}")


if __name__ == "__main__":
    main()
