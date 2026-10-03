"""Frozen E1-E3 methods on TARGET; all source files are read-only.

TPOT's four outer folds chose different pipelines and no final pipeline was saved.
The fold-0 pipeline is selected by fold index, without looking at its score or
TARGET, verified against its frozen held-out predictions, then refit on all 117.
This is explicitly a TPOT-derived fixed-pipeline sensitivity, not a new search.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFwe
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from tpot.builtin_modules import Passthrough, SkipTransformer
from xgboost import XGBClassifier


PACKAGE = Path(__file__).resolve().parents[1]
TRAIN = PACKAGE / "data" / "training"
METHOD = PACKAGE / "reference_outputs" / "method_comparison"
PRIOR = PACKAGE / "outputs" / "target_rf"
OUT = PACKAGE / "outputs" / "target_multimethod"
LABELS = ("E1", "E2", "E3")
METHODS = ("R2", "KS", "Wasserstein", "RF", "XGBoost", "LightGBM", "TPOT_fixed_fold0")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tpot_fold0_pipeline():
    """Exact untruncated pipeline_repr from frozen tpot_run_log.json fold 0."""
    passthrough_union = lambda: FeatureUnion(
        transformer_list=[("skiptransformer", SkipTransformer()),
                          ("passthrough", Passthrough())]
    )
    return Pipeline([
        ("standardscaler", StandardScaler()),
        ("selectfwe", SelectFwe(alpha=0.0030320892422)),
        ("featureunion-1", passthrough_union()),
        ("featureunion-2", passthrough_union()),
        ("sgdclassifier", SGDClassifier(
            alpha=4.60801815e-05, class_weight="balanced",
            eta0=0.183142172636, fit_intercept=True,
            l1_ratio=0.504911374881, learning_rate="optimal",
            loss="modified_huber", n_jobs=1, penalty="elasticnet",
            random_state=20260920,
        )),
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "train_index": TRAIN / "model_inputs/model_index.csv",
        "train_kde": TRAIN / "model_inputs/X_KDE_bw20_grid10.csv",
        "method_code": PACKAGE / "src" / "run_main_methods.py",
        "tpot_code": PACKAGE / "src" / "run_tpot.py",
        "tpot_log": METHOD / "tpot_run_log.json",
        "tpot_oof": METHOD / "tpot_oof_predictions.csv",
        "target_key": PRIOR / "TARGET_predictions_key.csv",
        "target_kde": PRIOR / "TARGET_X_KDE_bw20_grid10.csv",
        "target_index": PRIOR / "TARGET_feature_row_index.csv",
        "traditional_scores": PRIOR / "TARGET_traditional_reference_scores.csv",
        "rf_artifact": PRIOR / "RF_E1E3_full117_KDE_bw20_seed20260920.joblib",
    }
    source_hashes = {key: sha256(value) for key, value in files.items()}
    index = pd.read_csv(files["train_index"])
    train_x = pd.read_csv(files["train_kde"]).to_numpy(np.float32)
    target = pd.read_csv(files["target_key"])
    target_index = pd.read_csv(files["target_index"])
    target_x = pd.read_csv(files["target_kde"]).to_numpy(np.float32)
    assert len(index) == len(train_x) == 117
    assert index.E_label.value_counts().to_dict() == {"E2": 53, "E3": 34, "E1": 30}
    assert len(target) == len(target_index) == len(target_x) == 140
    assert target.sample_instance_uid.tolist() == target_index.sample_instance_uid.tolist()
    assert int(target.training_sample_overlap.sum()) == 10
    same_doi_only = target.training_study_doi_overlap & ~target.training_sample_overlap
    assert int(same_doi_only.sum()) == 17
    target["same_training_DOI_distinct_sample"] = same_doi_only
    target["analysis_scope"] = np.where(target.training_sample_overlap,
                                         "EXCLUDED_PHYSICAL_DUPLICATE", "INDEPENDENT_PHYSICAL_SAMPLE")
    y = index.E_label.map({label: i for i, label in enumerate(LABELS)}).to_numpy(int)

    # Existing RF predictions are authoritative; verify the saved final model,
    # rather than overwrite or replace previously published RF values.
    rf_bundle = joblib.load(files["rf_artifact"])
    assert rf_bundle["training_sample_uids"] == index.global_sample_uid.tolist()
    rf_prob = rf_bundle["model"].predict_proba(target_x)
    prior_rf = target[[f"RF_probability_{label}" for label in LABELS]].to_numpy(float)
    assert np.allclose(rf_prob, prior_rf, atol=1e-12)
    assert np.array_equal(rf_prob.argmax(axis=1), target.RF_predicted_class.map({l: i for i, l in enumerate(LABELS)}))

    counts = np.bincount(y, minlength=3)
    class_weights = len(y) / (3 * counts)
    w = class_weights[y]
    xgb = XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss",
        random_state=20260920, n_jobs=2, tree_method="hist",
    )
    lightgbm = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=5, colsample_bytree=0.8, subsample=0.8,
        subsample_freq=1, reg_lambda=1.0, random_state=20260920,
        n_jobs=2, verbosity=-1,
    )
    xgb.fit(train_x, y, sample_weight=w)
    lightgbm.fit(train_x, y, sample_weight=w)
    for name, model in (("XGBoost", xgb), ("LightGBM", lightgbm)):
        p = model.predict_proba(target_x)
        assert p.shape == (140, 3) and np.allclose(p.sum(axis=1), 1)
        target[f"{name}_predicted_class"] = [LABELS[i] for i in np.argmax(p, axis=1)]
        for j, label in enumerate(LABELS):
            target[f"{name}_probability_{label}"] = p[:, j]
        target[f"{name}_max_probability_uncalibrated"] = p.max(axis=1)
        joblib.dump({"model": model, "labels": LABELS,
                     "training_sample_uids": index.global_sample_uid.tolist(),
                     "class_sample_weights": class_weights.tolist(),
                     "parameters_fixed_from": str(files["method_code"])}, OUT / f"{name}_full117_fixed.joblib")

    # Reject a guessed TPOT pipeline: first reproduce the frozen fold-0 OOF
    # labels exactly on that fold's untouched study-level test samples.
    log = json.loads(files["tpot_log"].read_text(encoding="utf-8"))
    assert log["completed_folds"] == [0, 1, 2, 3]
    assert "SGDClassifier" in log["descriptors"][0]["pipeline_repr"]
    fold = index.test_fold.to_numpy(int)
    tr, te = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    validation_pipeline = tpot_fold0_pipeline()
    validation_pipeline.fit(train_x[tr], y[tr])
    validation_pred = validation_pipeline.predict(train_x[te]).astype(int)
    oof = pd.read_csv(files["tpot_oof"])
    frozen_fold0 = oof.loc[oof.fold.eq(0)].set_index("global_sample_uid")
    expected = frozen_fold0.loc[index.iloc[te].global_sample_uid, "predicted_label"].map(
        {l: i for i, l in enumerate(LABELS)}).to_numpy(int)
    agreement = int((validation_pred == expected).sum())
    assert agreement == len(te), f"TPOT fold-0 reconstruction mismatch: {agreement}/{len(te)}"
    pipeline = tpot_fold0_pipeline()
    pipeline.fit(train_x, y)
    tpot_pred = pipeline.predict(target_x).astype(int)
    target["TPOT_fixed_fold0_predicted_class"] = [LABELS[i] for i in tpot_pred]
    try:
        tpot_prob = pipeline.predict_proba(target_x)
        assert tpot_prob.shape == (140, 3)
        for j, label in enumerate(LABELS):
            target[f"TPOT_fixed_fold0_probability_{label}"] = tpot_prob[:, j]
        target["TPOT_fixed_fold0_max_probability_uncalibrated"] = tpot_prob.max(axis=1)
    except (AttributeError, NotImplementedError):
        pass
    joblib.dump({"model": pipeline, "labels": LABELS,
                 "training_sample_uids": index.global_sample_uid.tolist(),
                 "provenance": "TPOT outer-fold-0 selected pipeline, fixed before TARGET; reconstructed and refitted on 117",
                 "fold0_validation_n": len(te)}, OUT / "TPOT_fold0_frozen_pipeline_full117.joblib")

    # Preserve the already fixed traditional classifications and score values.
    score = pd.read_csv(files["traditional_scores"])
    assert len(score) == 420 and score.sample_instance_uid.nunique() == 140
    score_wide = score.pivot(index="sample_instance_uid", columns="reference_class",
                             values=["Pearson_R2_KDE", "KS_ECDF", "Wasserstein_1_Ma"])
    score_wide.columns = [f"{a}_{b}" for a, b in score_wide.columns]
    target = target.merge(score_wide.reset_index(), on="sample_instance_uid", validate="one_to_one")
    pred_columns = {
        "R2": "Pearson_R2_predicted_class", "KS": "KS_predicted_class",
        "Wasserstein": "Wasserstein_1_predicted_class",
        "RF": "RF_predicted_class", "XGBoost": "XGBoost_predicted_class",
        "LightGBM": "LightGBM_predicted_class",
        "TPOT_fixed_fold0": "TPOT_fixed_fold0_predicted_class",
    }
    independent = target.loc[~target.training_sample_overlap].copy()
    assert len(independent) == 130 and independent.sample_instance_uid.nunique() == 130
    pred = independent[list(pred_columns.values())].rename(columns={v: k for k, v in pred_columns.items()})
    votes = pred.apply(lambda row: row.value_counts(), axis=1).fillna(0).astype(int)
    for label in LABELS:
        independent[f"votes_{label}"] = votes[label].to_numpy(int)
    independent["max_vote_count"] = votes.max(axis=1).to_numpy(int)
    independent["n_distinct_predicted_classes"] = (votes > 0).sum(axis=1).to_numpy(int)
    independent["all_7_methods_agree"] = independent.max_vote_count.eq(7)
    independent["RF_vs_other_disagreement_count"] = sum(
        independent["RF_predicted_class"].ne(independent[column]).astype(int)
        for name, column in pred_columns.items() if name != "RF")
    independent["prediction_conflict_type"] = np.select(
        [independent.all_7_methods_agree,
         independent.max_vote_count.eq(6),
         independent.max_vote_count.ge(4)],
        ["UNANIMOUS_7", "ONE_DISSENTER", "MAJORITY_WITH_CONFLICT"],
        default="SPLIT_OR_TIE")
    # The TPOT search did not freeze a unique all-data pipeline. Keep a second
    # primary agreement definition limited to the six unambiguous fixed methods.
    core_methods = [m for m in METHODS if m != "TPOT_fixed_fold0"]
    core_votes = pred[core_methods].apply(lambda row: row.value_counts(), axis=1).fillna(0).astype(int)
    independent["max_vote_count_6_core"] = core_votes.max(axis=1).to_numpy(int)
    independent["all_6_core_methods_agree"] = independent.max_vote_count_6_core.eq(6)
    independent["n_distinct_6_core_classes"] = (core_votes > 0).sum(axis=1).to_numpy(int)
    method_matrix = []
    for a in METHODS:
        for b in METHODS:
            n = int((pred[a] == pred[b]).sum())
            method_matrix.append({"method_A": a, "method_B": b,
                                  "n_agree": n, "n_independent_targets": len(independent),
                                  "agreement_fraction": n / len(independent)})
    matrix = pd.DataFrame(method_matrix)
    core_matrix = matrix.loc[matrix.method_A.isin(core_methods) & matrix.method_B.isin(core_methods)].copy()
    external113 = independent.loc[~independent.same_training_DOI_distinct_sample]
    assert len(external113) == 113
    independent_study_matrix = pd.DataFrame([
        {"method_A": a, "method_B": b,
         "n_agree": int(external113[pred_columns[a]].eq(external113[pred_columns[b]]).sum()),
         "n_study_independent_targets": 113,
         "agreement_fraction": float(external113[pred_columns[a]].eq(external113[pred_columns[b]]).mean())}
        for a in METHODS for b in METHODS
    ])
    totals = pd.DataFrame([{
        "n_TARGET_source": 140, "n_excluded_physical_duplicate": 10,
        "n_independent_physical_TARGET": 130,
        "n_same_training_DOI_distinct_sample": int(independent.same_training_DOI_distinct_sample.sum()),
        "n_study_independent_TARGET": int((~independent.same_training_DOI_distinct_sample).sum()),
        "n_all_7_agree": int(independent.all_7_methods_agree.sum()),
        "n_any_conflict": int((~independent.all_7_methods_agree).sum()),
        "n_all_6_core_agree": int(independent.all_6_core_methods_agree.sum()),
        "n_any_6_core_conflict": int((~independent.all_6_core_methods_agree).sum()),
        "n_split_or_tie": int(independent.prediction_conflict_type.eq("SPLIT_OR_TIE").sum()),
    }])
    method_counts = []
    for method, column in pred_columns.items():
        for label in LABELS:
            method_counts.append({"method": method, "predicted_class": label,
                                  "n_independent_TARGET": int(independent[column].eq(label).sum())})
    for name, frame in (
        ("01_TARGET_all140_with_flags.csv", target),
        ("02_TARGET_independent130_multimethod.csv", independent),
        ("03_pairwise_agreement_long.csv", matrix),
        ("03b_pairwise_agreement_6_core_long.csv", core_matrix),
        ("03c_pairwise_agreement_study_independent113.csv", independent_study_matrix),
        ("04_conflict_samples.csv", independent.loc[~independent.all_7_methods_agree]),
        ("05_10_physical_duplicates_excluded.csv", target.loc[target.training_sample_overlap]),
        ("06_17_same_DOI_distinct_samples.csv", independent.loc[independent.same_training_DOI_distinct_sample]),
        ("07_method_class_counts.csv", pd.DataFrame(method_counts)),
        ("08_summary_counts.csv", totals),
    ):
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    provenance = {
        "n_training": 117, "training_labels": {l: int(counts[i]) for i, l in enumerate(LABELS)},
        "n_training_studies": int(index.cv_group.nunique()),
        "n_TARGET_source": 140, "n_TARGET_analysis": 130,
        "excluded_physical_duplicates": 10, "same_DOI_distinct_samples_flagged": 17,
        "RF_prior_probability_max_abs_difference": float(np.max(np.abs(rf_prob - prior_rf))),
        "TPOT_fixed_pipeline_origin": "outer fold 0 by prespecified fold index; NOT selected by performance or TARGET",
        "TPOT_fold0_reconstruction_match": f"{agreement}/{len(te)}",
        "TPOT_not_a_new_automl_search": True,
        "KDE": "0-4600 Ma, 10 Ma grid, Gaussian bandwidth 20 Ma, area normalized",
        "traditional_scores": "reused prior full-117 reference scores; class sample-equal weights",
        "probabilities": "uncalibrated classification probabilities, NOT provenance fractions",
        "source_sha256": source_hashes,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    assert {key: sha256(value) for key, value in files.items()} == source_hashes
    (OUT / "run_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(totals.to_string(index=False), flush=True)
    print(pd.DataFrame(method_counts).pivot(index="method", columns="predicted_class", values="n_independent_TARGET").to_string(), flush=True)
    print("TPOT fold0 reproduction:", agreement, "/", len(te), flush=True)


if __name__ == "__main__":
    main()
