"""Prespecified feature and grain-threshold sensitivity, preserving DOI groups."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from run_main_methods import LABELS, OUT, ROOT, fit_ml, metrics, summarize


SOURCE_BUILDER = ROOT / "scripts" / "build_training_inputs.py"
spec = importlib.util.spec_from_file_location("frozen_input_builder", SOURCE_BUILDER)
frozen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(frozen)
N20 = pd.read_csv(ROOT / "model_inputs" / "model_index.csv")
EXISTING_FOLDS = pd.read_csv(ROOT / "cv_group_assignments.csv").set_index("cv_group")["test_fold"].to_dict()


def derive_pool():
    meta = pd.read_csv(frozen.DB / "MASTER" / "all_samples_metadata_full.csv", low_memory=False)
    prior = pd.read_csv(frozen.PRIOR_AUDIT, low_memory=False)
    e1 = pd.read_csv(frozen.E1_AUDIT / "E1_retained_full_metadata.csv", low_memory=False)
    audited = prior.loc[(prior["endpoint"].isin(("E2", "E3"))) & (prior["decision"] == "KEEP_MAIN") & prior["material_audit"].isin(frozen.DETRITAL)].copy()
    e1_ids, audited_ids = set(e1["global_sample_uid"]), set(audited["global_sample_uid"])
    eligible_ids = e1_ids | audited_ids
    assert len(e1_ids) == 30 and len(audited_ids) == 97 and len(eligible_ids) == 127
    pool = meta.loc[meta["global_sample_uid"].isin(eligible_ids)].copy()
    material_map = audited.set_index("global_sample_uid")["material_audit"]
    pool["material_for_training"] = pool["global_sample_uid"].map(material_map).fillna(pool["material_group"])
    assert pool["material_for_training"].isin(frozen.DETRITAL).all()
    pool["E_label"] = pool["endpoint"]
    pool["cv_group"] = pool.apply(lambda r: frozen.normalized_group(r["doi"], r["study_id"]), axis=1)
    source = pd.read_csv(frozen.DB / "MASTER" / "all_endpoints_UPb_full.csv",
                         usecols=["global_sample_uid", "global_record_uid", "grain_id", "age_Ma", "analysis_use_flag", "doi"], low_memory=False)
    g = source.loc[source["global_sample_uid"].isin(eligible_ids) & source["analysis_use_flag"].isin(frozen.VALID_FLAGS)
                   & source["age_Ma"].notna() & (source["age_Ma"] > 0)].copy()
    author_accept = frozen.author_river_accept()
    river = g["doi"].astype(str).str.lower().eq("10.3389/feart.2020.568715")
    assert river.sum() == 1025
    g = g.loc[~river | g["grain_id"].astype(str).map(author_accept).eq(True)].copy()
    ages = {uid: np.sort(frame["age_Ma"].to_numpy(float)) for uid, frame in g.groupby("global_sample_uid")}
    pool["n_model_ages"] = pool["global_sample_uid"].map({k: len(v) for k, v in ages.items()}).fillna(0).astype(int)
    return pool, ages


def allocate_folds(current):
    assignments = dict(EXISTING_FOLDS)
    grouped = current.groupby(["E_label", "cv_group"], as_index=False).size().rename(columns={"size": "n_samples"})
    for label in LABELS:
        class_rows = grouped.loc[grouped["E_label"] == label]
        counts = np.zeros(4, dtype=int)
        for row in class_rows.itertuples(index=False):
            if row.cv_group in assignments:
                counts[assignments[row.cv_group]] += row.n_samples
        new = class_rows.loc[~class_rows["cv_group"].isin(assignments)].sort_values("cv_group")
        for row in new.itertuples(index=False):
            target = int(np.argmin(counts))
            assignments[row.cv_group] = target
            counts[target] += row.n_samples
    current = current.copy()
    current["test_fold"] = current["cv_group"].map(assignments).astype(int)
    assert current.groupby("cv_group")["test_fold"].nunique().max() == 1
    assert pd.crosstab(current["test_fold"], current["E_label"]).gt(0).all().all()
    return current


def features(current, ages, kind):
    arr = [ages[uid] for uid in current["global_sample_uid"]]
    if kind == "KDE_bw20":
        hist = np.zeros((len(arr), 4601), dtype=np.float32)
        for i, values in enumerate(arr):
            hist[i], _ = np.histogram(values, bins=np.arange(-0.5, 4601.5, 1.0))
        x = gaussian_filter1d(hist, sigma=20.0, axis=1, mode="constant", truncate=4.0)[:, ::10]
        x = x / np.trapezoid(x, x=frozen.KDE_GRID, axis=1)[:, None]
        return x
    if kind == "Piecewise_A":
        edges = frozen.PIECEWISE_EDGES
        x = np.empty((len(arr), len(edges) - 1), dtype=np.float32)
        for i, values in enumerate(arr):
            counts, _ = np.histogram(values, bins=edges)
            x[i] = counts / counts.sum()
        return x
    raise ValueError(kind)


def main():
    pool, ages = derive_pool()
    predictions, allocations, counts, fold_metric_rows = [], [], [], []
    for threshold in (10, 20, 30):
        current = allocate_folds(pool.loc[pool["n_model_ages"] >= threshold].copy())
        current = current.sort_values(["E_label", "cv_group", "sample_id", "global_sample_uid"]).reset_index(drop=True)
        assert current["global_sample_uid"].is_unique
        if threshold == 20:
            assert current["global_sample_uid"].tolist() == N20["global_sample_uid"].tolist()
            assert current["test_fold"].tolist() == N20["test_fold"].tolist()
        allocations.extend(current[["global_sample_uid", "sample_id", "E_label", "cv_group", "test_fold", "n_model_ages"]].assign(threshold=threshold).to_dict("records"))
        for label in LABELS:
            sub = current.loc[current["E_label"] == label]
            counts.append({"threshold": threshold, "E_label": label, "n_samples": len(sub), "n_studies": sub["cv_group"].nunique(), "n_grains": int(sub["n_model_ages"].sum())})
        y = current["E_label"].to_numpy()
        folds = current["test_fold"].to_numpy()
        for feature in ("KDE_bw20", "Piecewise_A"):
            x = features(current, ages, feature)
            if threshold == 20:
                name = "X_KDE_bw20_grid10.csv" if feature == "KDE_bw20" else "X_Piecewise_A.csv"
                baseline = pd.read_csv(ROOT / "model_inputs" / name).to_numpy(np.float32)
                assert np.allclose(x, baseline, atol=2e-5)
            for fold in range(4):
                train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                assert not set(current.iloc[train]["cv_group"]) & set(current.iloc[test]["cv_group"])
                for method in ("Random_Forest", "XGBoost", "LightGBM"):
                    predicted = fit_ml(method, train, test, y, x)
                    method_key = f"{method}__{feature}__n{threshold}"
                    for i, p in zip(test, predicted):
                        row = current.iloc[i]
                        predictions.append({"method": method_key, "classifier": method, "feature": feature, "threshold": threshold,
                                            "fold": fold, "global_sample_uid": row.global_sample_uid, "cv_group": row.cv_group,
                                            "true_label": row.E_label, "predicted_label": p})
                    print(f"n>={threshold} {feature} {method} fold {fold} complete", flush=True)
    pred = pd.DataFrame(predictions)
    overview, folds, confusion, studies = summarize(pred)
    key_parts = overview["method"].str.extract(r"^(.*?)__(.*?)__n(\d+)$")
    overview.insert(1, "classifier", key_parts[0])
    overview.insert(2, "feature", key_parts[1])
    overview.insert(3, "threshold", key_parts[2].astype(int))
    pd.DataFrame(allocations).to_csv(OUT / "sensitivity_sample_folds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(counts).to_csv(OUT / "sensitivity_threshold_counts.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / "sensitivity_oof_predictions.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(OUT / "sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "sensitivity_fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(OUT / "sensitivity_confusion_long.csv", index=False, encoding="utf-8-sig")
    studies.to_csv(OUT / "sensitivity_study_metrics.csv", index=False, encoding="utf-8-sig")
    print(overview[["classifier", "feature", "threshold", "macro_f1", "balanced_accuracy", "fold_sd_macro_f1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
