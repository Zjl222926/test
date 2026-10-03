"""Prespecified DOI-grouped E1-E3 method comparison; never reads TARGET."""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE / "data" / "training"
OUT = PACKAGE / "outputs" / "method_comparison_cpu"
LABELS = ("E1", "E2", "E3")
SEED = 20260920


def inputs():
    index = pd.read_csv(ROOT / "model_inputs" / "model_index.csv")
    x = pd.read_csv(ROOT / "model_inputs" / "X_KDE_bw20_grid10.csv").to_numpy(np.float32)
    with (ROOT / "model_inputs" / "age_arrays.jsonl").open(encoding="utf-8") as stream:
        arr = [json.loads(line) for line in stream]
    ages = [np.asarray(item["ages_Ma"], dtype=float) for item in arr]
    assert len(index) == len(x) == len(ages) == 117
    assert [item["global_sample_uid"] for item in arr] == index["global_sample_uid"].tolist()
    assert index["E_label"].value_counts().to_dict() == {"E2": 53, "E3": 34, "E1": 30}
    return index, x, ages


def reference_cdfs(train_ages: list[np.ndarray]):
    """Exact sample-equal mixture ECDF (not grain-count-weighted)."""
    values = np.concatenate(train_ages)
    weights = np.concatenate([np.full(len(a), 1.0 / (len(train_ages) * len(a))) for a in train_ages])
    order = np.argsort(values)
    return values[order], weights[order]


def weighted_ks(test: np.ndarray, ref_values: np.ndarray, ref_weights: np.ndarray):
    grid = np.unique(np.concatenate((test, ref_values)))
    sample_cdf = np.searchsorted(test, grid, side="right") / len(test)
    ref_cumsum = np.concatenate(([0.0], np.cumsum(ref_weights)))
    ref_cdf = ref_cumsum[np.searchsorted(ref_values, grid, side="right")]
    return float(np.max(np.abs(sample_cdf - ref_cdf)))


def traditional_fold(train, test, y, x, ages, fold):
    # Each class prototype is built exclusively from samples in the training fold.
    specs = {}
    for label in LABELS:
        loc = [i for i in train if y[i] == label]
        specs[label] = {
            "kde": x[loc].mean(axis=0),
            "cdf": reference_cdfs([ages[i] for i in loc]),
            "n_train_samples": len(loc),
        }
    pred = {"R2_KDE": [], "KS_ECDF": [], "Wasserstein_1": []}
    rows = []
    for i in test:
        test_age = ages[i]
        values = {}
        for label in LABELS:
            ref = specs[label]
            # Detrital-age-spectrum similarity R²: squared Pearson correlation
            # between the test KDE and train-only class reference KDE.
            pearson_r = float(np.corrcoef(x[i], ref["kde"])[0, 1])
            r2 = pearson_r * pearson_r
            # Store regression-style 1-SSE/SST separately because it is a
            # different quantity often also called R² and may be negative.
            denominator = np.sum((x[i] - x[i].mean()) ** 2)
            r2_sse = 1.0 - np.sum((x[i] - ref["kde"]) ** 2) / denominator
            ref_age, ref_weights = ref["cdf"]
            ks = weighted_ks(test_age, ref_age, ref_weights)
            w1 = wasserstein_distance(test_age, ref_age, v_weights=ref_weights)
            values[label] = (float(r2), float(ks), float(w1), float(r2_sse), pearson_r)
        r2_label = max(LABELS, key=lambda label: (values[label][0], -LABELS.index(label)))
        ks_label = min(LABELS, key=lambda label: (values[label][1], LABELS.index(label)))
        w1_label = min(LABELS, key=lambda label: (values[label][2], LABELS.index(label)))
        for method, label in (("R2_KDE", r2_label), ("KS_ECDF", ks_label), ("Wasserstein_1", w1_label)):
            pred[method].append(label)
        for label in LABELS:
            rows.append({"fold": fold, "test_row_index": int(i), "reference_class": label,
                         "R2_KDE": values[label][0], "KS_ECDF": values[label][1], "Wasserstein_1_Ma": values[label][2],
                         "R2_SSE_auxiliary": values[label][3], "Pearson_r_signed": values[label][4]})
    return pred, rows


def model_factory(name: str):
    if name == "Random_Forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                      max_features="sqrt", class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=2)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                             reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss",
                             random_state=SEED, n_jobs=2, tree_method="hist")
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15,
                              min_child_samples=5, colsample_bytree=0.8, subsample=0.8,
                              subsample_freq=1, reg_lambda=1.0, random_state=SEED,
                              n_jobs=2, verbosity=-1)
    raise ValueError(name)


def fit_ml(name, train, test, y, x):
    clf = model_factory(name)
    encoding = {label: i for i, label in enumerate(LABELS)}
    y_train = np.asarray([encoding[y[i]] for i in train])
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = len(train) / (3 * class_counts)
    if name == "Random_Forest":
        clf.fit(x[train], y_train)
    else:
        clf.fit(x[train], y_train, sample_weight=class_weights[y_train])
    y_hat = clf.predict(x[test]).astype(int)
    return [LABELS[i] for i in y_hat]


def metrics(y_true, y_pred):
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "n_test": len(y_true),
    }
    for i, label in enumerate(LABELS):
        result.update({f"{label}_precision": p[i], f"{label}_recall": r[i], f"{label}_f1": f[i], f"{label}_n": int(support[i])})
    return result


def summarize(pred: pd.DataFrame):
    fold_rows = []
    overall_rows = []
    confusion_rows = []
    study_rows = []
    for method, current in pred.groupby("method", sort=False):
        fold_values = []
        for fold, part in current.groupby("fold"):
            item = {"method": method, "fold": int(fold), **metrics(part["true_label"], part["predicted_label"])}
            fold_rows.append(item)
            fold_values.append(item)
        item = {"method": method, **metrics(current["true_label"], current["predicted_label"])}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "E1_recall", "E2_recall", "E3_recall"):
            item[f"fold_mean_{metric}"] = float(np.mean([f[metric] for f in fold_values]))
            item[f"fold_sd_{metric}"] = float(np.std([f[metric] for f in fold_values], ddof=1))
        per_study = current.groupby("cv_group").apply(
            lambda group: pd.Series({"n_samples": len(group), "study_accuracy": np.mean(group["true_label"] == group["predicted_label"]), "class": group["true_label"].iloc[0]}),
            include_groups=False,
        ).reset_index()
        per_study.insert(0, "method", method)
        study_rows.extend(per_study.to_dict("records"))
        item["study_equal_accuracy"] = float(per_study["study_accuracy"].mean())
        overall_rows.append(item)
        cm = confusion_matrix(current["true_label"], current["predicted_label"], labels=LABELS)
        for i, actual in enumerate(LABELS):
            for j, predicted in enumerate(LABELS):
                confusion_rows.append({"method": method, "true_label": actual, "predicted_label": predicted, "n": int(cm[i, j])})
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows), pd.DataFrame(confusion_rows), pd.DataFrame(study_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index, x, ages = inputs()
    y = index["E_label"].to_numpy()
    folds = index["test_fold"].to_numpy()
    assert set(folds) == {0, 1, 2, 3}
    predictions = []
    reference_rows = []
    timings = []
    for fold in range(4):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        train_groups = set(index.iloc[train]["cv_group"])
        test_groups = set(index.iloc[test]["cv_group"])
        assert not train_groups & test_groups
        started = time.monotonic()
        traditional, refs = traditional_fold(train, test, y, x, ages, fold)
        reference_rows.extend(refs)
        timings.append({"method": "traditional_three_methods", "fold": fold, "seconds": time.monotonic() - started})
        for method, labels in traditional.items():
            for i, predicted in zip(test, labels):
                row = index.iloc[i]
                predictions.append({"method": method, "fold": fold, "row_index": int(i), "global_sample_uid": row.global_sample_uid,
                                    "cv_group": row.cv_group, "true_label": row.E_label, "predicted_label": predicted})
        print(f"fold {fold}: traditional baselines complete", flush=True)
        for method in ("Random_Forest", "XGBoost", "LightGBM"):
            started = time.monotonic()
            labels = fit_ml(method, train, test, y, x)
            timings.append({"method": method, "fold": fold, "seconds": time.monotonic() - started})
            for i, predicted in zip(test, labels):
                row = index.iloc[i]
                predictions.append({"method": method, "fold": fold, "row_index": int(i), "global_sample_uid": row.global_sample_uid,
                                    "cv_group": row.cv_group, "true_label": row.E_label, "predicted_label": predicted})
            print(f"fold {fold}: {method} complete in {timings[-1]['seconds']:.1f}s", flush=True)
    pred = pd.DataFrame(predictions)
    assert pred.groupby("method").size().eq(117).all()
    summary, fold_metrics, confusion, study_metrics = summarize(pred)
    pred.to_csv(OUT / "main_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "main_method_comparison.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / "main_fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(OUT / "main_confusion_long.csv", index=False, encoding="utf-8-sig")
    study_metrics.to_csv(OUT / "main_study_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reference_rows).to_csv(OUT / "traditional_reference_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timings).to_csv(OUT / "runtime_seconds.csv", index=False, encoding="utf-8-sig")
    metadata = {"seed": SEED, "python": platform.python_version(), "methods": summary["method"].tolist(),
                "n_samples": 117, "n_groups": 21, "outer_folds": 4,
                "R2_definition": "Pearson correlation squared between 461-point test KDE and training-class sample-equal mean KDE; signed r and alternative 1-SSE/SST saved in traditional_reference_scores.csv",
                "KS_definition": "supremum absolute ECDF difference vs training-class sample-equal mixture ECDF",
                "Wasserstein_definition": "W1 in Ma vs training-class sample-equal weighted mixture of raw ages",
                "reference_training_only": True, "target_used": False,
                "fixed_ml_hyperparameters": {name: model_factory(name).get_params() for name in ("Random_Forest", "XGBoost", "LightGBM")}}
    (OUT / "main_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(summary[["method", "macro_f1", "balanced_accuracy", "E1_recall", "E2_recall", "E3_recall", "fold_sd_macro_f1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
