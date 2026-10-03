"""Read-only open-world detrital age-spectrum affinity test for frozen TARGET.

This is a descriptive reference comparison, not classification, provenance
unmixing, label correction, or model training. External reference samples from
the same DOI as each TARGET are left out of that TARGET's reference prototype.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wasserstein_distance


PACKAGE = Path(__file__).resolve().parents[1]
DB = PACKAGE / "data" / "full_database" / "MASTER"
TRAIN = PACKAGE / "data" / "training" / "model_inputs"
TARGET = PACKAGE / "outputs" / "target_multimethod"
PRIOR = PACKAGE / "outputs" / "target_rf"
E4_AUDIT = PACKAGE / "data" / "audits" / "01_E4_sample_audit_all_candidates.csv"
TARGET_GRAIN = PACKAGE / "data" / "target" / "TARGET_Borneo_detrital_zircon_MASTER_v2.csv"
OUT = PACKAGE / "outputs" / "open_world_affinity"
GRID = np.arange(0.0, 4601.0, 10.0)
MATERIALS = {"SEDIMENTARY_DETRITAL", "MODERN_RIVER_OR_SAND"}
GROUPS = ["C1", "C2", "C3", "E4_arc_analogue", "R3b_Cathaysia_margin",
          "R3b_Cathaysia_coast", "R3b_Hainan"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def age_signature(values: np.ndarray) -> str:
    a = np.sort(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_kde(age_lists: list[np.ndarray]) -> np.ndarray:
    hist = np.zeros((len(age_lists), 4601), dtype=np.float32)
    for i, values in enumerate(age_lists):
        assert len(values) >= 20 and np.isfinite(values).all() and ((values > 0) & (values <= 4600)).all()
        hist[i], _ = np.histogram(values, bins=np.arange(-0.5, 4601.5, 1.0))
        assert int(hist[i].sum()) == len(values)
    kde = gaussian_filter1d(hist, sigma=20.0, axis=1, mode="constant", truncate=4.0)[:, ::10]
    kde /= np.trapezoid(kde, x=GRID, axis=1)[:, None]
    assert kde.shape[1] == 461
    return kde


def reference_cdf(age_lists: list[np.ndarray]):
    values = np.concatenate(age_lists)
    weights = np.concatenate([np.full(len(a), 1.0 / (len(age_lists) * len(a))) for a in age_lists])
    order = np.argsort(values)
    return values[order], weights[order]


def ks_to_reference(test: np.ndarray, ref_values: np.ndarray, ref_weights: np.ndarray) -> float:
    grid = np.unique(np.concatenate((test, ref_values)))
    test_cdf = np.searchsorted(test, grid, side="right") / len(test)
    cumul = np.concatenate(([0.0], np.cumsum(ref_weights)))
    ref_cdf = cumul[np.searchsorted(ref_values, grid, side="right")]
    return float(np.max(np.abs(test_cdf - ref_cdf)))


def group_name(row):
    if row.endpoint in ("C1", "C2", "C3"):
        return row.endpoint
    if row.endpoint == "E4":
        return "E4_arc_analogue"
    component = str(row.source_component)
    if component == "E3_CATHAYSIA_HAINAN_TAIWAN_MARGIN":
        return "R3b_Cathaysia_margin"
    if component == "E3b_Cathaysia_coast_affinity_reference":
        return "R3b_Cathaysia_coast"
    if component == "E3b_Hainan_affinity_reference":
        return "R3b_Hainan"
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {
        "metadata": DB / "all_samples_metadata_full.csv",
        "grains": DB / "all_endpoints_UPb_full.csv",
        "e4_audit": E4_AUDIT,
        "training_index": TRAIN / "model_index.csv",
        "training_kde": TRAIN / "X_KDE_bw20_grid10.csv",
        "training_age_arrays": TRAIN / "age_arrays.jsonl",
        "target_prediction": TARGET / "02_TARGET_independent130_multimethod.csv",
        "target_kde": PRIOR / "TARGET_X_KDE_bw20_grid10.csv",
        "target_row_index": PRIOR / "TARGET_feature_row_index.csv",
        "target_grains": TARGET_GRAIN,
        "existing_scores": PRIOR / "TARGET_traditional_reference_scores.csv",
    }
    hashes = {k: digest(v) for k, v in inputs.items()}
    meta = pd.read_csv(inputs["metadata"], low_memory=False)
    upb = pd.read_csv(inputs["grains"], usecols=["global_sample_uid", "age_Ma", "analysis_use_flag"], low_memory=False)
    upb = upb.loc[upb.analysis_use_flag.isin(["VALID_AGE", "CANONICAL_VALID_AGE"])
                  & upb.age_Ma.gt(0) & upb.age_Ma.le(4600)]
    age_map = {uid: np.sort(part.age_Ma.to_numpy(float)) for uid, part in upb.groupby("global_sample_uid")}
    e4_audit = pd.read_csv(inputs["e4_audit"], low_memory=False)
    e4_audit = e4_audit.set_index("global_sample_uid", drop=False)
    train_index = pd.read_csv(inputs["training_index"])
    with inputs["training_age_arrays"].open(encoding="utf-8") as stream:
        train_age = [json.loads(line) for line in stream]
    assert len(train_index) == len(train_age) == 117
    train_kde = pd.read_csv(inputs["training_kde"]).to_numpy(np.float32)
    assert len(train_kde) == 117 and train_kde.shape[1] == 461
    assert np.allclose(make_kde([np.asarray(train_age[0]["ages_Ma"], dtype=float)])[0],
                       train_kde[0], atol=2e-5)
    train_signatures = {age_signature(np.asarray(row["ages_Ma"], dtype=float)) for row in train_age}
    train_uids = set(train_index.global_sample_uid)
    target = pd.read_csv(inputs["target_prediction"])
    target_row_index = pd.read_csv(inputs["target_row_index"])
    full_kde = pd.read_csv(inputs["target_kde"]).to_numpy(np.float32)
    assert len(target) == 130 and len(target_row_index) == len(full_kde) == 140
    kde_lookup = dict(zip(target_row_index.sample_instance_uid, full_kde))
    tg = pd.read_csv(inputs["target_grains"], usecols=["sample_instance_uid", "age_Ma", "doi"], low_memory=False)
    target_ages = {uid: np.sort(a[(a > 0) & (a <= 4600) & np.isfinite(a)])
                   for uid, part in tg.groupby("sample_instance_uid")
                   for a in [part.age_Ma.to_numpy(float)]}
    all_target_signatures = {age_signature(a) for a in target_ages.values()}
    assert set(target.sample_instance_uid).issubset(target_ages)
    assert int(target.same_training_DOI_distinct_sample.sum()) == 17
    assert len(target.loc[~target.same_training_DOI_distinct_sample]) == 113

    selected_meta = meta.loc[meta.endpoint.isin(["C1", "C2", "C3", "E4"]) | (
        meta.endpoint.eq("E3") & meta.source_component.astype(str).str.contains("CATHAYSIA|Cathaysia|Hainan", case=False))].copy()
    selected_meta["reference_group"] = [group_name(r) for r in selected_meta.itertuples(index=False)]
    selected_meta = selected_meta.loc[selected_meta.reference_group.notna()].copy()
    assert selected_meta.global_sample_uid.is_unique
    audit_rows = []
    for row in selected_meta.itertuples(index=False):
        uid = row.global_sample_uid
        ages = age_map.get(uid, np.array([], dtype=float))
        notes = []
        audit_material = str(row.material_group)
        primary_doi = str(row.doi) if pd.notna(row.doi) else ""
        if row.endpoint == "E4":
            if uid not in e4_audit.index:
                notes.append("E4_NOT_IN_GEOLOGIC_CANDIDATE_AUDIT")
            else:
                ev = e4_audit.loc[uid]
                audit_material = str(ev.material_group_audited)
                primary_doi = str(ev.verified_primary_doi) if pd.notna(ev.verified_primary_doi) else primary_doi
                if ev.physical_sample_status != "VALID_PHYSICAL_SAMPLE":
                    notes.append("E4_PHYSICAL_ID_NOT_VERIFIED")
                if "AFFINITY_REFERENCE" not in str(ev.recommended_dataset_role):
                    notes.append("E4_NOT_VERIFIED_ARC_RIVER_REFERENCE")
                if "PUBLISHED_SUB_MA_ZIRCONS_NOT_REPRODUCED" in str(ev.data_issue_flags):
                    notes.append("E4_PUBLISHED_YOUNG_COMPONENT_NOT_REPRODUCED")
        if audit_material not in MATERIALS:
            notes.append("NOT_DETRITAL_AFTER_MATERIAL_AUDIT")
        if len(ages) < 20:
            notes.append("N_LT20_VALID_AGES")
        if len(ages) != int(row.n_valid_ages):
            notes.append("METADATA_GRAIN_COUNT_MISMATCH")
        if not re.match(r"^10\.\d{4,9}/", primary_doi, re.IGNORECASE):
            notes.append("NO_VERIFIABLE_DOI")
        elif primary_doi.lower() == "10.5194/essd-18-3671-2026":
            notes.append("ONLY_ONEDZ_AGGREGATOR_DOI")
        sid = str(row.sample_id).strip() if pd.notna(row.sample_id) else ""
        if not sid or sid.lower() in ("nan", "none", "np", "unknown") or sid.isdecimal():
            notes.append("NO_SPECIFIC_PHYSICAL_SAMPLE_ID")
        if pd.isna(row.source_component):
            notes.append("MISSING_SOURCE_COMPONENT")
        if str(row.dataset_role) in ("DUPLICATE_VERSION_REFERENCE", "REVIEW_REQUIRED", "TARGET_OVERLAP_REFERENCE"):
            notes.append("ROLE_NOT_ELIGIBLE_FOR_REFERENCE")
        if str(row.cross_endpoint_exact_spectrum_flag) == "YES":
            notes.append("CROSS_ENDPOINT_EXACT_SPECTRUM")
        if str(row.model_leakage_guard) not in ("NONE", "nan", ""):
            notes.append("MODEL_LEAKAGE_GUARD")
        if uid in train_uids or (len(ages) >= 20 and age_signature(ages) in train_signatures):
            notes.append("EXACT_TRAINING_SPECTRUM_OR_UID")
        if len(ages) >= 20 and age_signature(ages) in all_target_signatures:
            notes.append("EXACT_TARGET_SPECTRUM_SELF_REFERENCE")
        audit_rows.append({
            "global_sample_uid": uid, "sample_instance_uid": row.sample_instance_uid,
            "sample_id": sid, "endpoint": row.endpoint, "reference_group": row.reference_group,
            "source_component": row.source_component, "dataset_role": row.dataset_role,
            "material_group_database": row.material_group, "material_group_audited": audit_material,
            "doi_database": row.doi, "doi_for_comparison": primary_doi,
            "study_id": row.study_id, "region": row.region, "formation": row.formation,
            "n_valid_ages_metadata": row.n_valid_ages, "n_valid_ages_recomputed": len(ages),
            "spectrum_fingerprint_database": row.spectrum_fingerprint,
            "age_signature_computed": age_signature(ages) if len(ages) else "",
            "cross_endpoint_exact_spectrum_flag": row.cross_endpoint_exact_spectrum_flag,
            "audit_exclusion_reasons": ";".join(notes),
            "reference_included": len(notes) == 0,
        })
    audit = pd.DataFrame(audit_rows)
    # If the same selected age spectrum is entered twice, retain no ambiguous
    # cross-domain copy. Same-domain copies have one deterministic representative.
    included = audit.loc[audit.reference_included].copy()
    for signature, part in included.groupby("age_signature_computed"):
        if len(part) <= 1:
            continue
        ordered = part.sort_values("global_sample_uid")
        drop = ordered.index if part.reference_group.nunique() > 1 else ordered.index[1:]
        audit.loc[drop, "reference_included"] = False
        audit.loc[drop, "audit_exclusion_reasons"] += ";DUPLICATE_SELECTED_SPECTRUM"
    selected = audit.loc[audit.reference_included].copy().reset_index(drop=True)
    assert selected.age_signature_computed.is_unique
    assert not set(selected.age_signature_computed) & (train_signatures | all_target_signatures)
    assert set(selected.reference_group) == set(GROUPS)
    selected["spectrum_index"] = np.arange(len(selected))
    ref_ages = [age_map[uid] for uid in selected.global_sample_uid]
    ref_kde = make_kde(ref_ages)
    ref_uid_to_pos = dict(zip(selected.global_sample_uid, selected.spectrum_index))
    assert len(ref_uid_to_pos) == len(selected)

    original_scores = pd.read_csv(inputs["existing_scores"])
    original_scores = original_scores.loc[original_scores.sample_instance_uid.isin(target.sample_instance_uid)]
    assert len(original_scores) == 390
    best_existing = original_scores.groupby("sample_instance_uid").agg(
        best_E_R2=("Pearson_R2_KDE", "max"),
        best_E_KS=("KS_ECDF", "min"),
        best_E_W1_Ma=("Wasserstein_1_Ma", "min"),
    )
    target = target.merge(best_existing.reset_index(), on="sample_instance_uid", validate="one_to_one")

    @lru_cache(maxsize=100)
    def reference(group: str, excluded_doi: str):
        part = selected.loc[selected.reference_group.eq(group) & ~selected.doi_for_comparison.str.lower().eq(excluded_doi)]
        if part.empty:
            return None
        positions = part.spectrum_index.to_numpy(int)
        kde = ref_kde[positions].mean(axis=0)
        values, weights = reference_cdf([ref_ages[i] for i in positions])
        return kde, values, weights, len(part), part.doi_for_comparison.nunique()

    rows = []
    for t in target.itertuples(index=False):
        uid = t.sample_instance_uid
        ages = target_ages[uid]
        x = kde_lookup[uid]
        assert len(ages) == int(t.n_grains_model)
        assert np.isclose(np.trapezoid(x, x=GRID), 1, atol=1e-5)
        doi = str(t.doi).lower().strip()
        for group in GROUPS:
            ref = reference(group, doi)
            if ref is None:
                rows.append({"sample_instance_uid": uid, "reference_group": group,
                             "reference_available_after_DOI_exclusion": False})
                continue
            mean_kde, rv, rw, n_refs, n_studies = ref
            r_signed = float(np.corrcoef(x, mean_kde)[0, 1])
            r2 = r_signed * r_signed
            ks = ks_to_reference(ages, rv, rw)
            w1 = float(wasserstein_distance(ages, rv, v_weights=rw))
            rows.append({
                "sample_instance_uid": uid, "reference_group": group,
                "reference_available_after_DOI_exclusion": True,
                "n_reference_samples_after_DOI_exclusion": n_refs,
                "n_reference_studies_after_DOI_exclusion": n_studies,
                "R2_KDE": r2, "Pearson_r_signed": r_signed,
                "KS_ECDF": ks, "Wasserstein_1_Ma": w1,
                "best_E_R2_KDE": t.best_E_R2, "best_E_KS_ECDF": t.best_E_KS,
                "best_E_Wasserstein_1_Ma": t.best_E_W1_Ma,
                "delta_R2_external_minus_best_E": r2 - t.best_E_R2,
                "delta_KS_best_E_minus_external": t.best_E_KS - ks,
                "delta_W1_best_E_minus_external_Ma": t.best_E_W1_Ma - w1,
                "reference_same_DOI_excluded": int(selected.loc[selected.reference_group.eq(group) & selected.doi_for_comparison.str.lower().eq(doi)].shape[0]),
            })
    long = pd.DataFrame(rows)
    long = long.merge(target[["sample_instance_uid", "sample_id", "doi", "region", "basin",
                              "depositional_period", "n_grains_model", "RF_predicted_class",
                              "RF_max_probability_uncalibrated", "RF_vs_other_disagreement_count",
                              "max_vote_count_6_core", "same_training_DOI_distinct_sample", "quality_flags"]],
                      on="sample_instance_uid", validate="many_to_one")
    long["R2_external_better"] = long.delta_R2_external_minus_best_E.gt(0)
    long["KS_external_better"] = long.delta_KS_best_E_minus_external.gt(0)
    long["W1_external_better"] = long.delta_W1_best_E_minus_external_Ma.gt(0)
    long["n_metrics_external_better"] = long[["R2_external_better", "KS_external_better", "W1_external_better"]].sum(axis=1)
    long["directional_3of3_external_preference"] = long.n_metrics_external_better.eq(3) & long.Pearson_r_signed.gt(0)
    long["analysis_priority_tags"] = [";".join(flags) if flags else "ROUTINE" for flags in (
        (["RF_E3"] if row.RF_predicted_class == "E3" else []) +
        (["RF_P_LT_0_5"] if row.RF_max_probability_uncalibrated < 0.5 else []) +
        (["METHOD_CONFLICT_GE3"] if row.RF_vs_other_disagreement_count >= 3 or row.max_vote_count_6_core <= 4 else [])
        for row in long.itertuples(index=False))]
    long["interpretation_limit"] = "Age-spectrum affinity only; not direct sediment supply or provenance fraction"

    summary_rows = []
    for scope, subset in (("nonphysical_overlap_130", long),
                          ("strict_cross_training_study_113", long.loc[~long.same_training_DOI_distinct_sample])):
        assert subset.sample_instance_uid.nunique() == (130 if scope.endswith("130") else 113)
        for group, part in subset.groupby("reference_group", sort=False):
            usable = part.loc[part.reference_available_after_DOI_exclusion]
            summary_rows.append({
                "scope": scope, "reference_group": group,
                "n_TARGET_total": subset.sample_instance_uid.nunique(),
                "n_TARGET_with_available_reference": len(usable),
                "n_REFERENCE_samples_total": int(selected.reference_group.eq(group).sum()),
                "n_REFERENCE_studies_total": int(selected.loc[selected.reference_group.eq(group), "doi_for_comparison"].nunique()),
                "n_R2_external_better": int(usable.R2_external_better.sum()),
                "n_KS_external_better": int(usable.KS_external_better.sum()),
                "n_W1_external_better": int(usable.W1_external_better.sum()),
                "n_directional_3of3": int(usable.directional_3of3_external_preference.sum()),
                "fraction_directional_3of3_of_available": float(usable.directional_3of3_external_preference.mean()) if len(usable) else np.nan,
                "median_delta_W1_Ma": float(usable.delta_W1_best_E_minus_external_Ma.median()) if len(usable) else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    candidates = long.loc[long.directional_3of3_external_preference
                          & long.n_reference_studies_after_DOI_exclusion.ge(2)].copy()
    candidates = candidates.sort_values(["sample_instance_uid", "delta_W1_best_E_minus_external_Ma"], ascending=[True, False])
    top_candidates = candidates.drop_duplicates("sample_instance_uid").copy()
    top_candidates["candidate_status"] = "AGE_SPECTRUM_SIMILARITY_ONLY_REQUIRES_GEOLOGIC_VALIDATION"
    top_candidates["review_priority_score"] = (
        2 * top_candidates.RF_predicted_class.eq("E3").astype(int)
        + 2 * top_candidates.RF_max_probability_uncalibrated.lt(0.5).astype(int)
        + top_candidates.RF_vs_other_disagreement_count.ge(3).astype(int)
    )
    top_candidates = top_candidates.sort_values(
        ["review_priority_score", "delta_W1_best_E_minus_external_Ma"], ascending=[False, False]
    ).reset_index(drop=True)
    one_row = target[["sample_instance_uid", "sample_id", "doi", "region", "basin",
                      "depositional_period", "n_grains_model", "RF_predicted_class",
                      "RF_max_probability_uncalibrated", "Pearson_R2_predicted_class",
                      "KS_predicted_class", "Wasserstein_1_predicted_class",
                      "RF_vs_other_disagreement_count", "same_training_DOI_distinct_sample",
                      "quality_flags", "best_E_R2", "best_E_KS", "best_E_W1_Ma"]].copy()
    for metric, ascending, prefix in (("R2_KDE", False, "R2"), ("KS_ECDF", True, "KS"),
                                      ("Wasserstein_1_Ma", True, "W1")):
        best = (long.loc[long.reference_available_after_DOI_exclusion]
                .sort_values(["sample_instance_uid", metric], ascending=[True, ascending])
                .drop_duplicates("sample_instance_uid")
                [["sample_instance_uid", "reference_group", metric]])
        best = best.rename(columns={"reference_group": f"best_external_{prefix}_group",
                                    metric: f"best_external_{prefix}_value"})
        one_row = one_row.merge(best, on="sample_instance_uid", validate="one_to_one")
    match_groups = (candidates.groupby("sample_instance_uid")["reference_group"]
                    .agg(lambda x: ";".join(sorted(set(x)))).rename("directional_3of3_external_groups"))
    one_row = one_row.merge(match_groups.reset_index(), on="sample_instance_uid", how="left", validate="one_to_one")
    one_row["directional_3of3_external_groups"] = one_row.directional_3of3_external_groups.fillna("")
    one_row["any_directional_3of3_external_preference"] = one_row.directional_3of3_external_groups.ne("")
    one_row["analysis_priority_tags"] = [";".join(flags) if flags else "ROUTINE" for flags in (
        (["RF_E3"] if row.RF_predicted_class == "E3" else []) +
        (["RF_P_LT_0_5"] if row.RF_max_probability_uncalibrated < 0.5 else []) +
        (["METHOD_CONFLICT_GE3"] if row.RF_vs_other_disagreement_count >= 3 else [])
        for row in one_row.itertuples(index=False))]
    bedrock_classes = {"IGNEOUS_OR_VOLCANIC_BEDROCK", "METAMORPHIC_BASEMENT_OR_METASEDIMENT"}
    e4_bedrock_corrected = {
        uid for uid, ev in e4_audit.iterrows()
        if str(ev.material_group_audited) in bedrock_classes
    }
    bedrock = selected_meta.loc[selected_meta.material_group.isin(bedrock_classes)
                                | selected_meta.global_sample_uid.isin(e4_bedrock_corrected)].copy()
    bedrock = bedrock[["global_sample_uid", "endpoint", "source_component", "sample_id", "doi",
                       "material_group", "region", "formation", "n_valid_ages"]]
    bedrock["audited_material_group"] = [str(e4_audit.loc[uid].material_group_audited)
                                         if uid in e4_bedrock_corrected else material
                                         for uid, material in zip(bedrock.global_sample_uid, bedrock.material_group)]
    bedrock["E4_audit_material_applied"] = bedrock.global_sample_uid.isin(e4_bedrock_corrected)
    bedrock["use_role"] = "GEOLOGIC_CONTEXT_ONLY_NOT_SAME_KIND_AGE_SPECTRUM_REFERENCE"
    audit.to_csv(OUT / "01_reference_identity_material_audit.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "02_included_detrital_reference_manifest.csv", index=False, encoding="utf-8-sig")
    bedrock.to_csv(OUT / "03_bedrock_geologic_context_only.csv", index=False, encoding="utf-8-sig")
    long.to_csv(OUT / "04_TARGET_external_affinity_long.csv", index=False, encoding="utf-8-sig")
    one_row.to_csv(OUT / "04b_TARGET_one_row_affinity_summary.csv", index=False, encoding="utf-8-sig")
    e_long = original_scores.rename(columns={"reference_class": "reference_group",
                                             "Pearson_R2_KDE": "R2_KDE",
                                             "Wasserstein_1_Ma": "Wasserstein_1_Ma"}).copy()
    e_long["reference_type"] = "FROZEN_TRAINING_ENDPOINT"
    ext_long = long[["sample_instance_uid", "reference_group", "R2_KDE", "KS_ECDF",
                     "Wasserstein_1_Ma", "Pearson_r_signed"]].copy()
    ext_long["reference_type"] = "EXTERNAL_DETRITAL_REFERENCE"
    all_scores = pd.concat([e_long[["sample_instance_uid", "reference_group", "reference_type",
                                    "R2_KDE", "KS_ECDF", "Wasserstein_1_Ma", "Pearson_r_signed"]],
                            ext_long[["sample_instance_uid", "reference_group", "reference_type",
                                      "R2_KDE", "KS_ECDF", "Wasserstein_1_Ma", "Pearson_r_signed"]]],
                           ignore_index=True)
    all_scores.to_csv(OUT / "04c_TARGET_all_E_and_external_reference_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "05_scope_summary_130_113.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT / "06_all_directional_3of3_matches.csv", index=False, encoding="utf-8-sig")
    top_candidates.to_csv(OUT / "07_potential_omitted_affinity_target_list.csv", index=False, encoding="utf-8-sig")
    # Save all reference prototypes on the exact 461-point KDE grid for audit/plots.
    prototypes = []
    for label in ("E1", "E2", "E3"):
        part = train_index.E_label.eq(label).to_numpy()
        mean_kde = train_kde[part].mean(axis=0)
        prototypes.extend({"reference_group": label, "reference_type": "FROZEN_TRAINING_ENDPOINT",
                           "age_Ma": age, "mean_KDE_density": float(density)}
                          for age, density in zip(GRID, mean_kde))
    for group in GROUPS:
        part = selected.loc[selected.reference_group.eq(group)]
        mean_kde = ref_kde[part.spectrum_index.to_numpy(int)].mean(axis=0)
        prototypes.extend({"reference_group": group, "reference_type": "EXTERNAL_DETRITAL_REFERENCE",
                           "age_Ma": age, "mean_KDE_density": float(density)}
                          for age, density in zip(GRID, mean_kde))
    pd.DataFrame(prototypes).to_csv(OUT / "08_E_and_external_KDE_prototypes.csv", index=False, encoding="utf-8-sig")
    summary_json = {
        "n_training": 117, "n_target_independent_physical": 130,
        "n_target_strict_cross_training_study": 113,
        "n_reference_detrital": len(selected),
        "n_reference_bedrock_context": len(bedrock),
        "n_reference_excluded_audit": int((~audit.reference_included).sum()),
        "n_target_with_directional_3of3_any_group": int(top_candidates.sample_instance_uid.nunique()),
        "method": "Equal-sample reference KDE/ECDF, exact same feature grid; per TARGET remove same DOI external samples; compare to existing frozen E1-E3 reference scores",
        "directional_preference_is_not_statistical_significance": True,
        "age_affinity_not_direct_supply": True,
        "source_sha256": hashes,
    }
    assert {k: digest(v) for k, v in inputs.items()} == hashes
    (OUT / "run_manifest.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("included", len(selected), "excluded", (~audit.reference_included).sum(),
          "top directional candidates", len(top_candidates), flush=True)


if __name__ == "__main__":
    main()
