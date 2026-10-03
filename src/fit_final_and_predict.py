"""One authorized final full-data RF fit and TARGET_STRATIGRAPHIC prediction.

The frozen training and source TARGET files are read-only. No labels, feature
definitions, classifier parameters, or probability thresholds are optimized on
TARGET. Cross-study OOF model-comparison results are not re-estimated here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier


PACKAGE = Path(__file__).resolve().parents[1]
TRAIN = PACKAGE / "data" / "training"
TARGET = PACKAGE / "data" / "target" / "TARGET_Borneo_detrital_zircon_MASTER_v2.csv"
OUT = PACKAGE / "outputs" / "target_rf"
LABELS = ("E1", "E2", "E3")
GRID = np.arange(0.0, 4601.0, 10.0)
SEED = 20260920
RF_PARAMS = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                 class_weight="balanced_subsample", random_state=SEED, n_jobs=2)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_key(doi: object, sample_id: object):
    doi_part = str(doi).lower().strip().replace("https://doi.org/", "").replace("doi:", "")
    sample_part = re.sub(r"[^a-z0-9]", "", str(sample_id).lower())
    return f"{doi_part}::{sample_part}"


def make_kde(age_lists):
    hist = np.zeros((len(age_lists), 4601), dtype=np.float32)
    for i, values in enumerate(age_lists):
        assert len(values) > 0 and np.isfinite(values).all() and ((values > 0) & (values <= 4600)).all()
        hist[i], _ = np.histogram(values, bins=np.arange(-0.5, 4601.5, 1.0))
        assert int(hist[i].sum()) == len(values)
    kde = gaussian_filter1d(hist, sigma=20.0, axis=1, mode="constant", truncate=4.0)[:, ::10]
    kde /= np.trapezoid(kde, x=GRID, axis=1)[:, None]
    assert kde.shape[1] == 461 and np.allclose(np.trapezoid(kde, x=GRID, axis=1), 1, atol=1e-5)
    return kde


def reference_cdf(train_age_lists):
    ages = np.concatenate(train_age_lists)
    weights = np.concatenate([np.full(len(a), 1.0 / (len(train_age_lists) * len(a))) for a in train_age_lists])
    order = np.argsort(ages)
    return ages[order], weights[order]


def ks_to_reference(test, ref_age, ref_weights):
    grid = np.unique(np.concatenate((test, ref_age)))
    test_cdf = np.searchsorted(test, grid, side="right") / len(test)
    ref_cumulative = np.concatenate(([0.0], np.cumsum(ref_weights)))
    reference_cdf_values = ref_cumulative[np.searchsorted(ref_age, grid, side="right")]
    return float(np.max(np.abs(test_cdf - reference_cdf_values)))


def nonblank(value):
    return pd.notna(value) and str(value).strip() != ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_index = pd.read_csv(TRAIN / "model_inputs" / "model_index.csv")
    train_features = pd.read_csv(TRAIN / "model_inputs" / "X_KDE_bw20_grid10.csv")
    train_x = train_features.to_numpy(np.float32)
    with (TRAIN / "model_inputs" / "age_arrays.jsonl").open(encoding="utf-8") as fh:
        age_lines = [json.loads(line) for line in fh]
    train_ages = [np.asarray(item["ages_Ma"], dtype=float) for item in age_lines]
    assert len(train_index) == len(train_x) == len(train_ages) == 117
    assert train_index["E_label"].value_counts().to_dict() == {"E2": 53, "E3": 34, "E1": 30}
    assert [item["global_sample_uid"] for item in age_lines] == train_index["global_sample_uid"].tolist()
    # Confirm the target and training feature construction are numerically identical.
    assert np.allclose(make_kde([train_ages[0]])[0], train_x[0], atol=2e-5)
    encoding = {label: i for i, label in enumerate(LABELS)}
    y = train_index["E_label"].map(encoding).to_numpy(int)

    grain = pd.read_csv(TARGET, low_memory=False)
    assert set(grain["target_type"]) == {"TARGET_STRATIGRAPHIC"}
    assert len(grain) == 14838 and grain["sample_instance_uid"].nunique() == 140
    assert grain["sample_instance_uid"].notna().all() and grain["age_Ma"].notna().all()
    assert grain.groupby("sample_instance_uid")["doi"].nunique().max() == 1
    assert grain.groupby("sample_instance_uid")["sample_id"].nunique().max() == 1
    target_meta = grain.drop_duplicates("sample_instance_uid").copy().reset_index(drop=True)
    target_meta = target_meta.drop(columns=["grain_id", "age_Ma", "age_1sigma_Ma", "concordance_pct"])
    target_meta = target_meta.sort_values(["doi", "sample_id", "sample_instance_uid"]).reset_index(drop=True)
    groups = {uid: frame["age_Ma"].to_numpy(float) for uid, frame in grain.groupby("sample_instance_uid", sort=False)}
    target_age_lists = []
    out_of_grid = []
    raw_counts = []
    for uid in target_meta["sample_instance_uid"]:
        raw = groups[uid]
        good = raw[(raw > 0) & (raw <= 4600) & np.isfinite(raw)]
        assert len(good) > 0
        target_age_lists.append(np.sort(good))
        raw_counts.append(len(raw))
        out_of_grid.append(len(raw) - len(good))
    target_meta["n_grains_source"] = raw_counts
    target_meta["n_grains_model"] = [len(values) for values in target_age_lists]
    target_meta["n_age_outside_0_4600_Ma"] = out_of_grid
    assert target_meta["n_grains_source"].sum() == 14838
    assert target_meta["n_grains_model"].sum() == 14837
    assert target_meta["n_age_outside_0_4600_Ma"].sum() == 1
    assert target_meta["n_grains_model"].lt(20).sum() == 1

    train_keys = [sample_key(doi, sid) for doi, sid in zip(train_index["cv_group"], train_index["sample_id"])]
    assert len(set(train_keys)) == 117
    lookup = dict(zip(train_keys, train_index["global_sample_uid"]))
    target_keys = [sample_key(doi, sid) for doi, sid in zip(target_meta["doi"], target_meta["sample_id"])]
    target_meta["matched_training_uid"] = [lookup.get(key, "") for key in target_keys]
    target_meta["training_sample_overlap"] = target_meta["matched_training_uid"].ne("")
    target_meta["training_study_doi_overlap"] = target_meta["doi"].astype(str).str.lower().isin(set(train_index["cv_group"]))
    target_meta["independent_sample_interpretation"] = ~target_meta["training_sample_overlap"]
    target_meta["study_independent_external"] = ~target_meta["training_study_doi_overlap"]
    assert target_meta["training_sample_overlap"].sum() == 10
    assert target_meta["training_study_doi_overlap"].sum() == 27
    assert target_meta["study_independent_external"].sum() == 113

    depositional_fields = ["depositional_age_text", "depositional_period", "depositional_age_min_Ma",
                           "depositional_age_max_Ma", "estimated_depositional_age_Ma"]
    target_meta["depositional_age_unspecified"] = ~target_meta[depositional_fields].apply(
        lambda r: any(nonblank(value) for value in r), axis=1)
    target_meta["estimated_depositional_age_Ma_missing"] = target_meta["estimated_depositional_age_Ma"].isna()
    target_meta["low_n_lt20"] = target_meta["n_grains_model"].lt(20)
    assert target_meta["depositional_age_unspecified"].sum() == 3
    flags = []
    for row in target_meta.itertuples(index=False):
        f = []
        if row.training_sample_overlap:
            f.append("TRAINING_SAMPLE_OVERLAP_NONINDEPENDENT")
        elif row.training_study_doi_overlap:
            f.append("SAME_STUDY_AS_TRAINING")
        if row.low_n_lt20:
            f.append("LOW_N_LT20")
        if row.depositional_age_unspecified:
            f.append("DEPOSITIONAL_AGE_UNSPECIFIED")
        if row.n_age_outside_0_4600_Ma:
            f.append("OUT_OF_GRID_AGE_OMITTED")
        flags.append(";".join(f) if f else "OK")
    target_meta["quality_flags"] = flags

    target_x = make_kde(target_age_lists)
    assert target_x.shape == (140, 461)
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(train_x, y)
    assert model.classes_.tolist() == [0, 1, 2]
    probabilities = model.predict_proba(target_x)
    assert probabilities.shape == (140, 3) and np.allclose(probabilities.sum(axis=1), 1.0)
    order = np.argsort(probabilities, axis=1)
    top = order[:, -1]
    top2 = order[:, -2]
    target_meta["RF_predicted_class"] = [LABELS[i] for i in top]
    for j, label in enumerate(LABELS):
        target_meta[f"RF_probability_{label}"] = probabilities[:, j]
    target_meta["RF_max_probability_uncalibrated"] = probabilities[np.arange(len(probabilities)), top]
    target_meta["RF_top2_probability_margin"] = probabilities[np.arange(len(probabilities)), top] - probabilities[np.arange(len(probabilities)), top2]
    target_meta["RF_normalized_entropy_uncertainty"] = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)), axis=1) / np.log(3)
    assert target_meta["RF_normalized_entropy_uncertainty"].between(0, 1).all()

    ref = {}
    for label in LABELS:
        positions = np.flatnonzero(train_index["E_label"].to_numpy() == label)
        ref[label] = {"kde": train_x[positions].mean(axis=0),
                      "raw": reference_cdf([train_ages[i] for i in positions])}
    score_rows = []
    method_predictions = {"Pearson_R2": [], "KS": [], "Wasserstein_1": []}
    for i, age in enumerate(target_age_lists):
        scores = {}
        for label in LABELS:
            r = float(np.corrcoef(target_x[i], ref[label]["kde"])[0, 1])
            ref_ages, ref_weights = ref[label]["raw"]
            scores[label] = (r * r, ks_to_reference(age, ref_ages, ref_weights),
                             float(wasserstein_distance(age, ref_ages, v_weights=ref_weights)))
            score_rows.append({"sample_instance_uid": target_meta.iloc[i]["sample_instance_uid"],
                               "reference_class": label, "Pearson_R2_KDE": scores[label][0],
                               "KS_ECDF": scores[label][1], "Wasserstein_1_Ma": scores[label][2],
                               "Pearson_r_signed": r})
        method_predictions["Pearson_R2"].append(max(LABELS, key=lambda l: (scores[l][0], -LABELS.index(l))))
        method_predictions["KS"].append(min(LABELS, key=lambda l: (scores[l][1], LABELS.index(l))))
        method_predictions["Wasserstein_1"].append(min(LABELS, key=lambda l: (scores[l][2], LABELS.index(l))))
    for method, pred in method_predictions.items():
        target_meta[f"{method}_predicted_class"] = pred
        target_meta[f"RF_{method}_agree"] = target_meta["RF_predicted_class"].eq(pred)
    target_meta["n_traditional_methods_agree_RF"] = sum(target_meta[f"RF_{m}_agree"].astype(int) for m in method_predictions)
    target_meta["all_four_methods_agree"] = target_meta["n_traditional_methods_agree_RF"].eq(3)

    # Predict every sample, but never count the ten physical training overlaps
    # as independent TARGET evidence. Same-study distinct samples stay flagged.
    target_meta.to_csv(OUT / "TARGET_predictions_full_metadata.csv", index=False, encoding="utf-8-sig")
    key_columns = ["sample_instance_uid", "sample_id", "study_id", "doi", "target_type", "region", "basin", "formation",
                   "depositional_age_text", "depositional_period", "depositional_age_min_Ma", "depositional_age_max_Ma",
                   "estimated_depositional_age_Ma", "n_grains_source", "n_grains_model", "RF_predicted_class",
                   "RF_probability_E1", "RF_probability_E2", "RF_probability_E3", "RF_max_probability_uncalibrated",
                   "RF_top2_probability_margin", "RF_normalized_entropy_uncertainty", "Pearson_R2_predicted_class",
                   "KS_predicted_class", "Wasserstein_1_predicted_class", "n_traditional_methods_agree_RF", "all_four_methods_agree",
                   "training_sample_overlap", "matched_training_uid", "training_study_doi_overlap", "independent_sample_interpretation",
                   "study_independent_external", "low_n_lt20", "depositional_age_unspecified", "estimated_depositional_age_Ma_missing",
                   "n_age_outside_0_4600_Ma", "quality_flags"]
    target_meta[key_columns].to_csv(OUT / "TARGET_predictions_key.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(score_rows).to_csv(OUT / "TARGET_traditional_reference_scores.csv", index=False, encoding="utf-8-sig")
    target_meta.loc[target_meta["training_sample_overlap"], key_columns].to_csv(OUT / "TARGET_10_training_overlap_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_x, columns=train_features.columns).to_csv(OUT / "TARGET_X_KDE_bw20_grid10.csv", index=False, float_format="%.9g")
    target_meta[["sample_instance_uid", "sample_id", "doi", "n_grains_model"]].to_csv(OUT / "TARGET_feature_row_index.csv", index=False, encoding="utf-8-sig")

    per_type = []
    for t in ("TARGET_MODERN", "TARGET_STRATIGRAPHIC"):
        part = target_meta.loc[target_meta["target_type"] == t]
        per_type.append({"target_type": t, "n_samples": len(part), "n_source_grains": int(part["n_grains_source"].sum()),
                         "n_model_grains": int(part["n_grains_model"].sum()), "n_studies": part["doi"].nunique(),
                         "n_training_sample_overlap": int(part["training_sample_overlap"].sum()),
                         "n_nonoverlap_samples": int(part["independent_sample_interpretation"].sum()),
                         "n_study_independent_external": int(part["study_independent_external"].sum()),
                         "n_low_n_lt20": int(part["low_n_lt20"].sum()),
                         "n_depositional_age_unspecified": int(part["depositional_age_unspecified"].sum())})
    pd.DataFrame(per_type).to_csv(OUT / "TARGET_type_summary.csv", index=False, encoding="utf-8-sig")
    class_summary = (target_meta.loc[~target_meta["training_sample_overlap"]]
                     .groupby(["target_type", "RF_predicted_class"], as_index=False)
                     .agg(n_samples=("sample_instance_uid", "size"),
                          mean_RF_max_probability=("RF_max_probability_uncalibrated", "mean"),
                          median_RF_max_probability=("RF_max_probability_uncalibrated", "median"),
                          mean_normalized_entropy=("RF_normalized_entropy_uncertainty", "mean"),
                          n_all_methods_agree=("all_four_methods_agree", "sum")))
    class_summary.to_csv(OUT / "TARGET_independent_RF_class_summary.csv", index=False, encoding="utf-8-sig")
    agreement = []
    for scope, mask in (("all_140", np.ones(len(target_meta), dtype=bool)),
                        ("nonoverlap_130", ~target_meta["training_sample_overlap"].to_numpy()),
                        ("study_independent_113", target_meta["study_independent_external"].to_numpy())):
        part = target_meta.loc[mask]
        for method in method_predictions:
            agreement.append({"scope": scope, "comparator": method, "n_samples": len(part),
                              "n_agree_RF": int(part[f"RF_{method}_agree"].sum()),
                              "agreement_fraction": float(part[f"RF_{method}_agree"].mean())})
        agreement.append({"scope": scope, "comparator": "all_three_traditional", "n_samples": len(part),
                          "n_agree_RF": int(part["all_four_methods_agree"].sum()),
                          "agreement_fraction": float(part["all_four_methods_agree"].mean())})
    pd.DataFrame(agreement).to_csv(OUT / "TARGET_method_agreement_summary.csv", index=False, encoding="utf-8-sig")

    artifact = {"model": model, "class_labels": LABELS, "feature_names": list(train_features.columns),
                "feature_spec": {"age_grid_Ma": GRID.tolist(), "gaussian_bandwidth_Ma": 20,
                                 "histogram_bin_width_Ma": 1, "kde_area_normalized": True},
                "training_sample_uids": train_index["global_sample_uid"].tolist(),
                "random_state": SEED, "RF_params": RF_PARAMS}
    model_path = OUT / "RF_E1E3_full117_KDE_bw20_seed20260920.joblib"
    joblib.dump(artifact, model_path, compress=3)
    reloaded = joblib.load(model_path)
    assert np.allclose(reloaded["model"].predict_proba(target_x), probabilities)
    metadata = {"training_rows": 117, "training_class_counts": train_index["E_label"].value_counts().to_dict(),
                "target_samples": len(target_meta), "target_source_grains": int(target_meta["n_grains_source"].sum()),
                "target_model_grains": int(target_meta["n_grains_model"].sum()),
                "training_sample_overlap_count": int(target_meta["training_sample_overlap"].sum()),
                "training_study_overlap_count": int(target_meta["training_study_doi_overlap"].sum()),
                "low_n_count": int(target_meta["low_n_lt20"].sum()),
                "depositional_age_unspecified_count": int(target_meta["depositional_age_unspecified"].sum()),
                "target_source_sha256": digest(TARGET),
                "training_model_index_sha256": digest(TRAIN / "model_inputs" / "model_index.csv"),
                "training_KDE_sha256": digest(TRAIN / "model_inputs" / "X_KDE_bw20_grid10.csv"),
                "training_age_arrays_sha256": digest(TRAIN / "model_inputs" / "age_arrays.jsonl"),
                "model_joblib_sha256": digest(model_path), "RF_params": RF_PARAMS,
                "probabilities_calibrated": False, "target_used_for_fitting_or_parameter_selection": False,
                "traditional_reference_weighting": "equal weight per training sample within E class",
                "source_database_modified": False}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"target_samples": 140, "nonoverlap": 130, "same_study": 27,
                      "low_n": metadata["low_n_count"], "missing_deposition": metadata["depositional_age_unspecified_count"],
                      "RF_class_counts_nonoverlap": class_summary[["RF_predicted_class", "n_samples"]].to_dict("records")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
