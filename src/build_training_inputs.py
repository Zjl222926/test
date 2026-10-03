"""Build a read-only E1–E3 detrital training manifest and shared CV inputs.

This script never edits the frozen database or any preceding audit.  The
inclusion decisions are geological/source decisions, not model results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(r"C:\Users\admin\Desktop\新")
DB = ROOT / "outputs" / "Borneo_DZ_database_FINAL_v1_2026-09-14"
ZIP = ROOT / "outputs" / "Borneo_DZ_database_FINAL_v1_2026-09-14.zip"
E1_AUDIT = ROOT / "outputs" / "E1_Schwaner_affinity_reaudit_2026-09-20"
PRIOR_AUDIT = ROOT / "borneo_E1_E3_prefreeze_audit_2026-09-20" / "derived_data" / "E1_E3_sample_audit.csv"
RIVER_SOURCE = Path(r"C:\Users\admin\Desktop\数据库\端元E1\_source_data") / "Breitfeld2020_Frontiers568715_datasheet3_SuppTable3_riversand.xlsx"
OUT = ROOT / "outputs" / "Borneo_E1E3_3class_main_n20_2026-09-20"
INPUTS = OUT / "model_inputs"
SEED = 20260920
N_SPLITS = 4
LABELS = ("E1", "E2", "E3")
DETRITAL = ("SEDIMENTARY_DETRITAL", "MODERN_RIVER_OR_SAND")
VALID_FLAGS = ("VALID_AGE", "CANONICAL_VALID_AGE")
KDE_GRID = np.arange(0.0, 4601.0, 10.0)
LINEAR_EDGES = np.arange(0.0, 4600.0 + 25.0, 25.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def piecewise_edges() -> np.ndarray:
    edges = [0.0]
    for lo, hi, step in ((0, 300, 25), (300, 1000, 50), (1000, 2500, 100), (2500, 4600, 200)):
        for value in np.arange(lo, hi, step, dtype=float):
            if value > edges[-1] + 1e-9:
                edges.append(float(value))
        if hi > edges[-1] + 1e-9:
            edges.append(float(hi))
    result = np.asarray(edges)
    assert len(result) == 53 and result[-1] == 4600.0
    return result


PIECEWISE_EDGES = piecewise_edges()


def author_river_accept() -> dict[str, bool]:
    source = pd.read_excel(RIVER_SOURCE, header=None)
    mask = source[18].astype(str).str.upper().isin(("Y", "N"))
    data = source.loc[mask, [0, 18]].copy()
    data.columns = ["grain_id", "accept"]
    assert len(data) == 1025 and data["accept"].eq("Y").sum() == 837
    assert data["grain_id"].is_unique
    return dict(zip(data["grain_id"].astype(str), data["accept"].eq("Y")))


def group_folds(manifest: pd.DataFrame) -> pd.DataFrame:
    """Deterministic label-stratified study allocation, independent of model outcomes."""
    by_study = (
        manifest.groupby(["E_label", "cv_group"], as_index=False)
        .agg(n_samples=("global_sample_uid", "size"), n_ages=("n_model_ages", "sum"))
    )
    assert by_study.groupby("cv_group")["E_label"].nunique().max() == 1
    result = []
    total_samples = np.zeros(N_SPLITS, dtype=int)
    for label in LABELS:
        current = by_study.loc[by_study["E_label"] == label].copy()
        assert len(current) >= N_SPLITS, f"{label} has fewer studies than folds"
        current["tie"] = current["cv_group"].map(
            lambda group: hashlib.sha256(f"{SEED}|{label}|{group}".encode()).hexdigest()
        )
        current = current.sort_values(["n_samples", "tie"], ascending=[False, True])
        class_samples = np.zeros(N_SPLITS, dtype=int)
        for row in current.itertuples(index=False):
            fold = min(range(N_SPLITS), key=lambda f: (class_samples[f], total_samples[f], f))
            result.append({
                "cv_group": row.cv_group,
                "E_label": label,
                "test_fold": fold,
                "n_samples": int(row.n_samples),
                "n_ages": int(row.n_ages),
            })
            class_samples[fold] += int(row.n_samples)
            total_samples[fold] += int(row.n_samples)
    allocation = pd.DataFrame(result).sort_values(["test_fold", "E_label", "cv_group"])
    assert allocation["cv_group"].is_unique
    assert (pd.crosstab(allocation["test_fold"], allocation["E_label"]) > 0).all().all()
    return allocation


def normalized_group(doi: object, study_id: object) -> str:
    value = str(doi).strip().lower() if pd.notna(doi) and str(doi).strip() else ""
    if value and value != "nan":
        return value.replace("https://doi.org/", "").replace("doi:", "")
    fallback = str(study_id).strip().lower() if pd.notna(study_id) else ""
    if not fallback or fallback == "nan":
        raise ValueError("Sample lacks DOI and study identity")
    return f"study:{fallback}"


def write_features(ages_by_uid: dict[str, np.ndarray], index: pd.DataFrame) -> dict:
    arrays = [ages_by_uid[uid] for uid in index["global_sample_uid"]]
    hist = np.zeros((len(arrays), 4601), dtype=np.float32)
    for i, ages in enumerate(arrays):
        assert np.isfinite(ages).all() and ((ages >= 0) & (ages <= 4600)).all()
        hist[i], _ = np.histogram(ages, bins=np.arange(-0.5, 4601.5, 1.0))
    kde = gaussian_filter1d(hist, sigma=20.0, axis=1, mode="constant", truncate=4.0)[:, ::10]
    area = np.trapezoid(kde, x=KDE_GRID, axis=1)
    kde = kde / area[:, None]
    assert np.allclose(np.trapezoid(kde, x=KDE_GRID, axis=1), 1.0, atol=1e-5)
    kde_names = [f"kde_age_{int(x):04d}_Ma" for x in KDE_GRID]
    pd.DataFrame(kde, columns=kde_names).to_csv(INPUTS / "X_KDE_bw20_grid10.csv", index=False, float_format="%.9g")

    feature_spec = {
        "row_order_file": "model_index.csv",
        "KDE": {"file": "X_KDE_bw20_grid10.csv", "n_features": len(kde_names), "grid_Ma": KDE_GRID.tolist(), "Gaussian_bandwidth_Ma": 20, "area_normalized": True},
    }
    for label, edges, filename in (
        ("Piecewise_A", PIECEWISE_EDGES, "X_Piecewise_A.csv"),
        ("LinearBin_25Ma", LINEAR_EDGES, "X_LinearBin_25Ma.csv"),
    ):
        matrix = np.zeros((len(arrays), len(edges) - 1), dtype=np.float32)
        for i, ages in enumerate(arrays):
            counts, _ = np.histogram(ages, bins=edges)
            assert counts.sum() == len(ages)
            matrix[i] = counts / counts.sum()
        assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6)
        cols = [f"{label}_{int(lo):04d}_{int(hi):04d}_Ma" for lo, hi in zip(edges[:-1], edges[1:])]
        pd.DataFrame(matrix, columns=cols).to_csv(INPUTS / filename, index=False, float_format="%.9g")
        feature_spec[label] = {"file": filename, "n_features": len(cols), "bin_edges_Ma": edges.tolist(), "proportions_sum_to_one": True}
    (INPUTS / "feature_spec.json").write_text(json.dumps(feature_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return feature_spec


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(DB / "MASTER" / "all_samples_metadata_full.csv", low_memory=False)
    prior = pd.read_csv(PRIOR_AUDIT, low_memory=False)
    e1_review = pd.read_csv(E1_AUDIT / "E1_retained_full_metadata.csv", low_memory=False)
    e1_ids = set(e1_review["global_sample_uid"])
    assert len(e1_ids) == 30

    audited_23 = prior.loc[
        prior["endpoint"].isin(("E2", "E3"))
        & (prior["decision"] == "KEEP_MAIN")
        & prior["material_audit"].isin(DETRITAL)
    ].copy()
    ids_23 = set(audited_23["global_sample_uid"])
    assert len(ids_23) == len(audited_23) == 97
    pool_ids = e1_ids | ids_23
    assert len(pool_ids) == 127

    fields = [
        "global_record_uid", "global_sample_uid", "endpoint", "sample_instance_uid", "sample_id",
        "grain_id", "doi", "study_id", "age_Ma", "age_1sigma_Ma", "concordance_pct",
        "analysis_use_flag", "analysis_method", "source_file", "source_sheet", "source_url",
    ]
    grain_all = pd.read_csv(DB / "MASTER" / "all_endpoints_UPb_full.csv", usecols=fields, low_memory=False)
    grains = grain_all.loc[
        grain_all["global_sample_uid"].isin(pool_ids)
        & grain_all["analysis_use_flag"].isin(VALID_FLAGS)
        & grain_all["age_Ma"].notna()
        & (grain_all["age_Ma"] > 0)
    ].copy()
    assert grains["global_record_uid"].is_unique
    source_accept = author_river_accept()
    river_mask = grains["doi"].astype(str).str.lower().eq("10.3389/feart.2020.568715")
    assert int(river_mask.sum()) == 1025
    assert set(grains.loc[river_mask, "grain_id"].astype(str)) == set(source_accept)
    grains["author_accept_applied"] = pd.NA
    grains.loc[river_mask, "author_accept_applied"] = grains.loc[river_mask, "grain_id"].astype(str).map(source_accept)
    raw_counts = grains.groupby("global_sample_uid").size().rename("n_db_valid_age_rows")
    grains = grains.loc[~river_mask | grains["author_accept_applied"].eq(True)].copy()
    assert int((grains["doi"].astype(str).str.lower() == "10.3389/feart.2020.568715").sum()) == 837
    model_counts = grains.groupby("global_sample_uid").size().rename("n_model_ages")

    # Audit-proven material correction is retained as a separate field; source material is untouched.
    audit_by_uid = audited_23.set_index("global_sample_uid")
    pool = meta.loc[meta["global_sample_uid"].isin(pool_ids)].copy()
    assert len(pool) == 127 and pool["global_sample_uid"].is_unique
    pool["E_label"] = pool["endpoint"]
    pool["material_for_training"] = pool["material_group"]
    pool["cv_group_original"] = pool["cv_group"]
    pool["material_audit_flag"] = ""
    pool["material_audit_evidence"] = ""
    m23 = pool["global_sample_uid"].isin(ids_23)
    pool.loc[m23, "material_for_training"] = pool.loc[m23, "global_sample_uid"].map(audit_by_uid["material_audit"])
    pool.loc[m23, "material_audit_flag"] = pool.loc[m23, "global_sample_uid"].map(audit_by_uid["material_flag"]).fillna("")
    pool.loc[m23, "material_audit_evidence"] = pool.loc[m23, "global_sample_uid"].map(audit_by_uid["sample_material_grain"]).fillna("")
    pool["geology_audit_decision"] = "E1_RETAIN_AFFINITY"
    pool.loc[m23, "geology_audit_decision"] = "E2E3_KEEP_MAIN"
    pool["n_db_valid_age_rows"] = pool["global_sample_uid"].map(raw_counts).fillna(0).astype(int)
    pool["n_model_ages"] = pool["global_sample_uid"].map(model_counts).fillna(0).astype(int)
    pool["cv_group"] = pool.apply(lambda r: normalized_group(r["doi"], r["study_id"]), axis=1)
    assert pool["material_for_training"].isin(DETRITAL).all()
    assert (pool["n_model_ages"] == pool["global_sample_uid"].map(model_counts).fillna(0)).all()
    # The prior geological audit's corrected E3 MIN19 count is checked, not silently replaced.
    for row in audited_23.itertuples(index=False):
        assert int(model_counts.loc[row.global_sample_uid]) == int(row.n_upb_author_qc), row.sample_id

    pool["include_main_n20"] = pool["n_model_ages"] >= 20
    manifest = pool.loc[pool["include_main_n20"]].copy()
    assert manifest.groupby("E_label").size().to_dict() == {"E1": 30, "E2": 53, "E3": 34}
    assert manifest["sample_instance_uid"].is_unique
    assert manifest["spectrum_fingerprint"].is_unique
    assert manifest["n_model_ages"].sum() == 11637
    manifest = manifest.sort_values(["E_label", "cv_group", "sample_id", "global_sample_uid"]).reset_index(drop=True)

    allocation = group_folds(manifest)
    folds = allocation.set_index("cv_group")["test_fold"]
    manifest["test_fold"] = manifest["cv_group"].map(folds).astype(int)
    assert manifest["test_fold"].notna().all()
    assert pd.crosstab(manifest["test_fold"], manifest["E_label"]).gt(0).all().all()

    # Filter grain rows once more to the final n>=20 sample set.
    grains = grains.loc[grains["global_sample_uid"].isin(manifest["global_sample_uid"])].copy()
    assert len(grains) == 11637
    lookup = manifest.set_index("global_sample_uid")
    grains["E_label"] = grains["global_sample_uid"].map(lookup["E_label"])
    grains["cv_group"] = grains["global_sample_uid"].map(lookup["cv_group"])
    grains["test_fold"] = grains["global_sample_uid"].map(lookup["test_fold"])
    grains = grains.sort_values(["E_label", "cv_group", "global_sample_uid", "global_record_uid"])
    assert grains["global_record_uid"].is_unique

    manifest.to_csv(OUT / "final_training_manifest_full.csv", index=False, encoding="utf-8-sig")
    key_cols = [
        "global_sample_uid", "sample_instance_uid", "sample_id", "E_label", "study_id", "doi", "cv_group",
        "test_fold", "material_group", "material_for_training", "material_audit_flag", "material_audit_evidence", "source_component", "formation",
        "latitude", "longitude", "n_db_valid_age_rows", "n_model_ages", "geology_audit_decision",
    ]
    manifest[key_cols].to_csv(OUT / "final_training_manifest_key.csv", index=False, encoding="utf-8-sig")
    grains.to_csv(INPUTS / "model_input_grains.csv", index=False, encoding="utf-8-sig")
    allocation.to_csv(OUT / "cv_group_assignments.csv", index=False, encoding="utf-8-sig")
    manifest[["global_sample_uid", "sample_id", "E_label", "cv_group", "test_fold", "n_model_ages"]].to_csv(
        OUT / "cv_sample_assignments.csv", index=False, encoding="utf-8-sig"
    )
    split_rows = []
    for fold in range(N_SPLITS):
        for row in manifest.itertuples(index=False):
            split_rows.append({"fold": fold, "split": "test" if row.test_fold == fold else "train", "global_sample_uid": row.global_sample_uid, "E_label": row.E_label, "cv_group": row.cv_group})
    pd.DataFrame(split_rows).to_csv(OUT / "cv_train_test_rows.csv", index=False, encoding="utf-8-sig")

    index_cols = ["global_sample_uid", "sample_instance_uid", "sample_id", "E_label", "cv_group", "test_fold", "n_model_ages", "material_for_training"]
    model_index = manifest[index_cols].copy().reset_index(drop=True)
    model_index.insert(0, "row_index_zero_based", np.arange(len(model_index), dtype=int))
    model_index.to_csv(INPUTS / "model_index.csv", index=False, encoding="utf-8-sig")
    ages_by_uid = {
        uid: np.sort(group["age_Ma"].to_numpy(dtype=float))
        for uid, group in grains.groupby("global_sample_uid")
    }
    assert len(ages_by_uid) == len(model_index)
    with (INPUTS / "age_arrays.jsonl").open("w", encoding="utf-8") as stream:
        for row in model_index.itertuples(index=False):
            ages = ages_by_uid[row.global_sample_uid]
            assert len(ages) == row.n_model_ages
            stream.write(json.dumps({"row_index_zero_based": row.row_index_zero_based, "global_sample_uid": row.global_sample_uid, "ages_Ma": ages.tolist()}, ensure_ascii=False) + "\n")
    features = write_features(ages_by_uid, model_index)

    summary_rows = []
    for label in LABELS:
        d = manifest.loc[manifest["E_label"] == label]
        summary_rows.append({
            "E_label": label,
            "physical_samples": len(d),
            "independent_cv_groups": d["cv_group"].nunique(),
            "raw_db_valid_age_rows": int(d["n_db_valid_age_rows"].sum()),
            "model_input_grains": int(d["n_model_ages"].sum()),
            "sedimentary_detrital_samples": int(d["material_for_training"].eq("SEDIMENTARY_DETRITAL").sum()),
            "modern_river_sand_samples": int(d["material_for_training"].eq("MODERN_RIVER_OR_SAND").sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "class_sample_study_grain_summary.csv", index=False, encoding="utf-8-sig")
    fold_summary = manifest.groupby(["test_fold", "E_label"], as_index=False).agg(
        test_samples=("global_sample_uid", "size"),
        test_studies=("cv_group", "nunique"),
        test_grains=("n_model_ages", "sum"),
    )
    fold_summary.to_csv(OUT / "cv_fold_class_counts.csv", index=False, encoding="utf-8-sig")

    # Explain exclusions from the original MAIN candidate pool, including source-review and n threshold.
    original = meta.loc[
        meta["endpoint"].isin(LABELS) & meta["dataset_role"].eq("MAIN_TRAIN_CANDIDATE")
    ].copy()
    prior_decision = prior.drop_duplicates("global_sample_uid").set_index("global_sample_uid")
    e1_decision = pd.read_csv(E1_AUDIT / "derived_data" / "E1_all_39_candidate_audit.csv", low_memory=False).set_index("global_sample_uid")
    selected_ids = set(manifest["global_sample_uid"])
    original["included_main_n20"] = original["global_sample_uid"].isin(selected_ids)
    original["decision_source"] = np.where(original["endpoint"].eq("E1"), "E1_Schwaner_affinity_reaudit_2026-09-20", "E1_E3_prefreeze_audit_2026-09-20")
    original["geology_decision"] = original["global_sample_uid"].map(prior_decision["decision"])
    is_e1 = original["endpoint"].eq("E1")
    original.loc[is_e1, "geology_decision"] = original.loc[is_e1, "global_sample_uid"].map(e1_decision["audit_decision"])
    original["n_model_ages_if_audited"] = original["global_sample_uid"].map(model_counts)
    original["exclusion_reason"] = ""
    external_e1 = is_e1 & ~original["included_main_n20"]
    original.loc[external_e1, "exclusion_reason"] = original.loc[external_e1, "global_sample_uid"].map(e1_decision["audit_reason"])
    external_23 = (~is_e1) & ~original["geology_decision"].eq("KEEP_MAIN")
    original.loc[external_23, "exclusion_reason"] = original.loc[external_23, "global_sample_uid"].map(prior_decision["geology_basis"]).fillna("来源/身份审计未批准进入主训练")
    low_n = (~is_e1) & original["geology_decision"].eq("KEEP_MAIN") & ~original["included_main_n20"]
    original.loc[low_n, "exclusion_reason"] = "地质审计可保留，但作者/模型可用 U–Pb 年龄少于 20 粒；本次 n≥20 主分析不纳入"
    original[["global_sample_uid", "sample_instance_uid", "sample_id", "endpoint", "study_id", "doi", "source_component", "dataset_role", "material_group", "n_valid_ages", "included_main_n20", "decision_source", "geology_decision", "n_model_ages_if_audited", "exclusion_reason"]].to_csv(
        OUT / "selection_audit_original_main_candidates.csv", index=False, encoding="utf-8-sig"
    )

    checks = {
        "source_zip_sha256": sha256(ZIP),
        "e1_audit_full_csv_sha256": sha256(E1_AUDIT / "E1_retained_full_metadata.csv"),
        "e2e3_audit_csv_sha256": sha256(PRIOR_AUDIT),
        "river_author_source_sha256": sha256(RIVER_SOURCE),
        "classes": LABELS,
        "n_samples": len(manifest),
        "n_studies": manifest["cv_group"].nunique(),
        "n_model_grains": len(grains),
        "n_folds": N_SPLITS,
        "cv_seed": SEED,
        "sample_uid_unique": bool(manifest["sample_instance_uid"].is_unique),
        "spectrum_fingerprint_unique": bool(manifest["spectrum_fingerprint"].is_unique),
        "grain_record_uid_unique": bool(grains["global_record_uid"].is_unique),
        "all_fold_tests_have_all_three_classes": bool(pd.crosstab(manifest["test_fold"], manifest["E_label"]).gt(0).all().all()),
        "no_cross_fold_study_split": bool(all(manifest.groupby("cv_group")["test_fold"].nunique() == 1)),
        "n_external_target_or_control_included": int(manifest["endpoint"].isin(("E4", "C1", "C2", "C3", "TARGET")).sum()),
        "n_bedrock_included": int((~manifest["material_for_training"].isin(DETRITAL)).sum()),
        "n_cathaysia_hainan_included": int(manifest["source_component"].astype(str).str.contains("Cathaysia|Hainan|coast_affinity", case=False).sum()),
        "features": features,
    }
    assert checks["n_samples"] == 117 and checks["n_model_grains"] == 11637
    assert checks["no_cross_fold_study_split"] and checks["all_fold_tests_have_all_three_classes"]
    assert checks["n_external_target_or_control_included"] == checks["n_bedrock_included"] == checks["n_cathaysia_hainan_included"] == 0
    (OUT / "validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    def clean_rows(frame: pd.DataFrame) -> list[list[object]]:
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.values.tolist()

    workbook_payload = {
        "summary": {"columns": summary.columns.tolist(), "rows": clean_rows(summary)},
        "folds": {"columns": fold_summary.columns.tolist(), "rows": clean_rows(fold_summary)},
        "studies": {"columns": allocation.columns.tolist(), "rows": clean_rows(allocation)},
        "manifest": {"columns": key_cols, "rows": clean_rows(manifest[key_cols])},
        "checks": checks,
    }
    (OUT / "workbook_payload.json").write_text(json.dumps(workbook_payload, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print("TOTAL", len(manifest), manifest["cv_group"].nunique(), len(grains))
    print("FOLDS\n", pd.crosstab(manifest["test_fold"], manifest["E_label"]).to_string())


if __name__ == "__main__":
    main()
