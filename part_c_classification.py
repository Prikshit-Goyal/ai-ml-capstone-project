"""Part C - classify which parcels will miss their promise and pick the alert threshold.

Run from the repository root (after part_a_clean_eda.py):  python part_c_classification.py
Writes outputs/part_c_{models,feature_scores,selectk,tuning,thresholds,final_test,final_config}.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

TARGET = "is_delayed"
SEED = 42
TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
SELECT_KS = [5, 7, 10, 12, 17]
FINAL_K = 7
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_RECALL = 0.80
# Tie-break order for choosing the final configuration.
TIE_ORDER = ["default_forest_same_7_features", "grid_search", "randomized_search"]


def classification_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    os.makedirs("outputs", exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    X = table[FEATURES]
    y = table[TARGET]

    # 1. Stratified 70 / 15 / 15 split (same settings as Part B, plus stratify).
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"split {len(X_train)}/{len(X_val)}/{len(X_test)}, "
          f"delayed {y_train.sum()}/{y_val.sum()}/{y_test.sum()}")

    # 2. Blind baseline and three classifiers at the default 0.5 rule, on validation.
    model_rows = [{"model": "blind_all_on_time", **classification_row(y_val, np.zeros(len(y_val), dtype=int))}]

    logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    model_rows.append({"model": "logistic_regression", **classification_row(y_val, logreg.predict(X_val))})

    tree_f1 = {}
    for d in TREE_DEPTHS:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=SEED).fit(X_train, y_train)
        tree_f1[d] = f1_score(y_val, tree.predict(X_val))
    best_d = max(TREE_DEPTHS, key=lambda d: (tree_f1[d], -d))  # highest F1, ties -> smaller depth
    tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=SEED).fit(X_train, y_train)
    model_rows.append({"model": f"decision_tree_depth_{best_d}", **classification_row(y_val, tree.predict(X_val))})

    forest = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)
    model_rows.append({"model": "random_forest_default", **classification_row(y_val, forest.predict(X_val))})
    pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

    # 3. Which columns carry signal: forest importance and F-score on training rows.
    selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": selector_all.scores_,
        "rf_importance": forest.feature_importances_,
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
    })
    scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
    scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

    selectk_rows = []
    for k in SELECT_KS:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ]).fit(X_train, y_train)
        selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
    pd.DataFrame(selectk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

    # 4. Tuning on the 7 selected columns (selector fitted on training rows only; test untouched).
    selector = SelectKBest(f_classif, k=FINAL_K).fit(X_train, y_train)
    X_train_k = selector.transform(X_train)
    X_val_k = selector.transform(X_val)

    grid = GridSearchCV(RandomForestClassifier(random_state=SEED), PARAM_GRID, cv=5).fit(X_train_k, y_train)
    rand = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED), PARAM_GRID, n_iter=9, cv=5, random_state=SEED
    ).fit(X_train_k, y_train)
    default_k = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train_k, y_train)

    chosen_params = {
        "grid_search": grid.best_params_,
        "randomized_search": rand.best_params_,
        "default_forest_same_7_features": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    }
    tuning = pd.DataFrame([
        {"search": "grid_search", "candidates": len(grid.cv_results_["params"]), "cv_folds": 5,
         "total_fits": len(grid.cv_results_["params"]) * 5, "best_params": str(grid.best_params_),
         "best_cv_accuracy": grid.best_score_, "val_f1": f1_score(y_val, grid.predict(X_val_k))},
        {"search": "randomized_search", "candidates": len(rand.cv_results_["params"]), "cv_folds": 5,
         "total_fits": len(rand.cv_results_["params"]) * 5, "best_params": str(rand.best_params_),
         "best_cv_accuracy": rand.best_score_, "val_f1": f1_score(y_val, rand.predict(X_val_k))},
        {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
         "best_params": "defaults", "best_cv_accuracy": np.nan, "val_f1": f1_score(y_val, default_k.predict(X_val_k))},
    ])
    tuning.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

    # 5. Final configuration: highest val_f1, ties broken by TIE_ORDER.
    val_f1 = dict(zip(tuning["search"], tuning["val_f1"]))
    chosen_from = max(TIE_ORDER, key=lambda s: (val_f1[s], -TIE_ORDER.index(s)))
    params = chosen_params[chosen_from]
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=FINAL_K)),
        ("model", RandomForestClassifier(random_state=SEED, **params)),
    ]).fit(X_train, y_train)

    # 6. Alert threshold from validation probabilities.
    val_proba = final_model.predict_proba(X_val)[:, 1]
    threshold_rows = [
        {"threshold": t, **classification_row(y_val, (val_proba >= t).astype(int))} for t in THRESHOLDS
    ]
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
    meets_rule = thresholds.loc[thresholds["recall"] >= MIN_RECALL, "threshold"]
    threshold = float(meets_rule.max()) if len(meets_rule) else 0.1

    # 7. One test evaluation at the chosen threshold.
    test_proba = final_model.predict_proba(X_test)[:, 1]
    final_test = {"threshold": threshold, **classification_row(y_test, (test_proba >= threshold).astype(int))}
    pd.DataFrame([final_test]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)

    pd.DataFrame([{
        "selector_k": FINAL_K,
        "n_estimators": params["n_estimators"],
        "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "threshold": threshold,
        "chosen_from": chosen_from,
    }]).to_csv("outputs/part_c_final_config.csv", index=False)

    print(f"best tree depth {best_d}; final config from {chosen_from} {params}; threshold {threshold}; "
          f"test precision {final_test['precision']:.4f} recall {final_test['recall']:.4f} f1 {final_test['f1']:.4f}")


if __name__ == "__main__":
    main()
