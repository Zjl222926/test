"""Read-only E1 geological re-audit of the frozen candidate database.

No source CSV/ZIP is changed.  This script emits derived review tables only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\admin\Desktop\新")
SRC_DIR = ROOT / "outputs" / "Borneo_DZ_database_FINAL_v1_2026-09-14"
SRC_ZIP = ROOT / "outputs" / "Borneo_DZ_database_FINAL_v1_2026-09-14.zip"
OUT = ROOT / "outputs" / "E1_Schwaner_affinity_reaudit_2026-09-20"
DERIVED = OUT / "derived_data"
SOURCE_RIVER = Path(
    r"C:\Users\admin\Desktop\数据库\端元E1\_source_data"
) / "Breitfeld2020_Frontiers568715_datasheet3_SuppTable3_riversand.xlsx"

EVIDENCE = {
    "10.1016/j.jseaes.2022.105111": "https://www.sciencedirect.com/science/article/abs/pii/S1367912022000347",
    "10.1007/s00343-021-0405-6": "https://doi.org/10.1007/s00343-021-0405-6",
    "10.3389/feart.2020.568715": "https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.568715/full",
    "10.1130/g50966.1": "https://doi.org/10.1130/G50966.1",
    "10.1016/j.gr.2018.06.001": "https://www.sciencedirect.com/science/article/abs/pii/S1342937X18301588",
    "10.1029/2025gc012837": "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GC012837",
    "10.1016/j.lithos.2016.05.003": "https://www.sciencedirect.com/science/article/abs/pii/S0024493716300706",
    "10.1016/j.lithos.2021.106425": "https://research.monash.edu/en/publications/cretaceous-kuching-accretionary-orogenesis-in-malaysia-sarawak-ge/",
}

NGILI = {"STB75", "STB75b", "STB77", "TB60a"}
SERABANG = {"SR09", "SR30"}
LUPAR = {"17MY-60A1", "17MY-61C1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decision(sample_id: str, doi: str, formation: str) -> tuple[str, str, str, str]:
    """Status, audit-formation, reason, evidence type; never uses age spectrum."""
    if sample_id in NGILI:
        return (
            "EXTERNAL_REFERENCE",
            formation,
            "原文将 Ngili Sandstone 的主要物源解释为邻近 Sadong/Kuching 地层；未确立 Schwaner/SW Borneo 输出联系。保留数据，但不作为 E1 训练。",
            "地层、沉积体系和原文物源解释",
        )
    if sample_id in SERABANG:
        return (
            "EXTERNAL_REFERENCE",
            formation,
            "Serabang 属 West Sarawak 早期活动陆缘；该研究重点检验 Hainan–Borneo 亲缘，未给出本样品直接 Schwaner 输出证据。留作外部构造参照。",
            "地理构造单元及原文研究设计",
        )
    if sample_id in LUPAR:
        return (
            "EXTERNAL_REFERENCE",
            formation,
            "Lupar/Lubok Antu 构造混杂岩中的 greywacke；原文将其置于 Kuching 增生/海沟–弧系统并强调东马来半岛—越南大陆边缘亲缘，不能直接代表 Schwaner 输出。",
            "构造环境及原文物源解释",
        )
    if sample_id == "IK22-8":
        return (
            "EXTERNAL_REFERENCE",
            formation,
            "Manunggul Formation 采自婆罗洲东南部 Meratus 含金刚石沉积体系；原文讨论澳大利亚/Gondwana 来源，不属于 SW Borneo/Schwaner 集水或沉积系统。",
            "采样位置、地层及原文地质背景",
        )
    if doi.lower() == "10.3389/feart.2020.568715":
        return (
            "RETAIN_E1",
            formation,
            "现代河砂采自 Schwaner 山区河流。原始附表逐项确认 9 个不同样品编号/坐标，尽管正文误写 8 个；保留样品身份并另记作者接受颗粒数。",
            "原始补充表样品编号、坐标及流域",
        )
    if sample_id == "STB74b":
        return (
            "RETAIN_E1",
            "Bako–Mintu Sandstone",
            "数据库地层空白；Breitfeld & Hall 2018 正文 7.2.3 明确将 STB74b 列为 Bako–Mintu Sandstone（Gunung Ngili 上部转石），非 Ngili Sandstone。该单元与 Kayan/Schwaner 输出有地质联系。",
            "原文样品段落及地层纠正",
        )
    if doi.lower() == "10.1016/j.gr.2018.06.001":
        return (
            "RETAIN_E1",
            formation,
            "Kayan/Bako–Mintu/Silantek/Tutoop 沉积体系；原文解释存在 SW Borneo/Schwaner 供源或继承再循环。允许混合来源，不要求纯 Schwaner。",
            "地层、沉积体系和原文物源解释",
        )
    if doi.lower() == "10.1016/j.jseaes.2022.105111":
        return (
            "RETAIN_E1",
            formation,
            "Western Kuching–South Schwaner 侏罗纪沉积/火山碎屑体系；研究在构造与区域供源框架中联结西南婆罗洲岩浆弧。Bengkayang 样品亦按混合区域输出保留。",
            "采样区、地层及原文构造解释",
        )
    if doi.lower() == "10.1130/g50966.1":
        return (
            "RETAIN_E1",
            formation,
            "Quek et al. 2023 以 SW Borneo 碎屑锆石记录讨论块体历史；West Borneo 边缘样品保留为 SW Borneo affinity 的混合碎屑输出。",
            "原文研究区域与构造单元",
        )
    if doi.lower() == "10.1007/s00343-021-0405-6":
        return (
            "RETAIN_E1",
            formation,
            "Ketapang／Schwaner 地区碎屑样品，具有直接的西南婆罗洲地理和构造联系。",
            "采样地点及地质单元",
        )
    raise ValueError(f"Unreviewed candidate: {sample_id}, {doi}")


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SRC_DIR / "MASTER" / "all_samples_metadata_full.csv", low_memory=False)
    grains = pd.read_csv(SRC_DIR / "MASTER" / "all_endpoints_UPb_full.csv", low_memory=False)
    candidates = meta.loc[
        (meta["endpoint"] == "E1")
        & (meta["dataset_role"] == "MAIN_TRAIN_CANDIDATE")
        & (meta["material_group"].isin(["SEDIMENTARY_DETRITAL", "MODERN_RIVER_OR_SAND"]))
    ].copy()
    assert len(candidates) == 39
    assert candidates["global_sample_uid"].is_unique
    assert candidates["sample_instance_uid"].is_unique

    full_e1 = meta.loc[meta["endpoint"] == "E1"].copy()
    same_fingerprint = full_e1.loc[full_e1["spectrum_fingerprint"].isin(candidates["spectrum_fingerprint"])]
    assert len(same_fingerprint) == len(candidates)
    assert candidates["spectrum_fingerprint"].is_unique
    all_fingerprint_matches = meta.loc[meta["spectrum_fingerprint"].isin(candidates["spectrum_fingerprint"])]
    all_uid_matches = meta.loc[meta["sample_instance_uid"].isin(candidates["sample_instance_uid"])]
    assert len(all_fingerprint_matches) == len(candidates)
    assert len(all_uid_matches) == len(candidates)

    source_sheet = pd.read_excel(SOURCE_RIVER, header=None)
    source_headers = source_sheet.loc[source_sheet[1].astype(str).str.startswith("("), [0, 1]].copy()
    source_headers.columns = ["sample_id", "source_coordinate_text"]
    assert set(source_headers["sample_id"]) == set(
        candidates.loc[candidates["doi"].str.lower() == "10.3389/feart.2020.568715", "sample_id"]
    )
    assert len(source_headers) == 9
    author_flag = source_sheet[18].astype(str).str.upper()
    assert author_flag.isin(["Y", "N"]).sum() == 1025
    assert author_flag.eq("Y").sum() == 837
    source_headers.to_csv(DERIVED / "Breitfeld2020_original_sample_ids.csv", index=False, encoding="utf-8-sig")

    outcomes = candidates.apply(
        lambda r: decision(str(r["sample_id"]), str(r["doi"]), str(r["formation"]) if pd.notna(r["formation"]) else ""),
        axis=1,
        result_type="expand",
    )
    outcomes.columns = ["audit_decision", "formation_audit", "audit_reason", "basis_type"]
    audit = pd.concat([candidates.reset_index(drop=True), outcomes.reset_index(drop=True)], axis=1)
    audit["source_evidence_url"] = audit["doi"].str.lower().map(EVIDENCE)
    audit["exact_duplicate_within_original_E1"] = False
    audit["exact_duplicate_elsewhere_in_current_db"] = False
    audit["uid_collision_elsewhere_in_current_db"] = False
    audit["identity_confirmed"] = True
    audit["source_identity_note"] = "DOI/study + original sample ID + unique sample UID + grain records"
    audit.loc[audit["sample_id"] == "STB74b", "source_identity_note"] = (
        "原文 7.2.3 明确列名；数据库 formation 字段空白，审计纠正为 Bako–Mintu，不改原数据库"
    )
    audit.loc[audit["doi"].str.lower() == "10.3389/feart.2020.568715", "source_identity_note"] = (
        "作者附表列有 9 个独立编号/坐标，正文写 8 个；1025 行均可追溯"
    )
    age_counts = grains.loc[
        grains["global_sample_uid"].isin(audit["global_sample_uid"])
        & (grains["analysis_use_flag"] == "VALID_AGE")
    ].groupby("global_sample_uid").size()
    audit["n_grain_rows_recounted"] = audit["global_sample_uid"].map(age_counts).fillna(0).astype(int)
    assert (audit["n_grain_rows_recounted"] == audit["n_valid_ages"]).all()
    audit["n_author_accepted_known"] = pd.NA
    river_mask = audit["doi"].str.lower() == "10.3389/feart.2020.568715"
    # Source-sheet accepted counts; source rows are grouped under each physical sample header.
    source_accept = {}
    current = None
    for row in source_sheet.itertuples(index=False, name=None):
        if isinstance(row[1], str) and row[1].startswith("("):
            current = row[0]
            source_accept[current] = 0
        if current is not None and str(row[18]).upper() == "Y":
            source_accept[current] += 1
    assert sum(source_accept.values()) == 837
    audit.loc[river_mask, "n_author_accepted_known"] = audit.loc[river_mask, "sample_id"].map(source_accept)
    audit["n_count_convention"] = "数据库 VALID_AGE 行；不是跨研究统一谐和度筛选后数"
    audit.loc[river_mask, "n_count_convention"] = "数据库 VALID_AGE 行；另列原文 accept=Y 颗粒数"

    keep = audit.loc[audit["audit_decision"] == "RETAIN_E1"].copy()
    not_keep = audit.loc[audit["audit_decision"] != "RETAIN_E1"].copy()
    assert len(keep) == 30 and len(not_keep) == 9
    assert keep["n_valid_ages"].sum() == 3174
    assert keep["doi"].str.lower().nunique() == 5
    assert keep["sample_instance_uid"].is_unique
    assert keep["spectrum_fingerprint"].is_unique

    audit.to_csv(DERIVED / "E1_all_39_candidate_audit.csv", index=False, encoding="utf-8-sig")
    keep.to_csv(OUT / "E1_retained_full_metadata.csv", index=False, encoding="utf-8-sig")
    not_keep.to_csv(OUT / "E1_external_or_deferred_full_metadata.csv", index=False, encoding="utf-8-sig")

    study_rows = []
    for doi, group in audit.groupby("doi", sort=False):
        retained = group.loc[group["audit_decision"] == "RETAIN_E1"]
        study_rows.append(
            {
                "doi": doi,
                "study_id": group["study_id"].iloc[0],
                "original_candidate_samples": len(group),
                "retained_samples": len(retained),
                "external_samples": len(group) - len(retained),
                "original_valid_age_rows": int(group["n_valid_ages"].sum()),
                "retained_valid_age_rows": int(retained["n_valid_ages"].sum()),
                "retained_sample_ids": ", ".join(retained["sample_id"]),
                "source_evidence_url": EVIDENCE[doi.lower()],
            }
        )
    study = pd.DataFrame(study_rows)
    study.to_csv(DERIVED / "E1_study_summary.csv", index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame(
        [
            {
                "definition": "原数据库全部 E1 metadata 行（含床岩和 TARGET_OVERLAP 参照；非训练数）",
                "physical_samples_or_rows": len(full_e1),
                "independent_studies": full_e1["doi"].str.lower().nunique(),
                "valid_age_rows": int(full_e1["n_valid_ages"].sum()),
                "note": "此行不可与碎屑训练样品直接等同；148 床岩/辅助 UID 行不能全视为独立物理样品。",
            },
            {
                "definition": "原 E1 碎屑 MAIN_TRAIN_CANDIDATE",
                "physical_samples_or_rows": len(candidates),
                "independent_studies": candidates["doi"].str.lower().nunique(),
                "valid_age_rows": int(candidates["n_valid_ages"].sum()),
                "note": "原数据库标记；含 9 个经地质审计转为外部参照的样品。",
            },
            {
                "definition": "本次统一 E1 = SW Borneo / Schwaner affinity 碎屑训练端元",
                "physical_samples_or_rows": len(keep),
                "independent_studies": keep["doi"].str.lower().nunique(),
                "valid_age_rows": int(keep["n_valid_ages"].sum()),
                "note": "不设 core/expanded；无 n 门槛；Breitfeld2020 1025 行中 837 为原文 accept=Y。",
            },
            {
                "definition": "转外部参照/暂不进入 E1",
                "physical_samples_or_rows": len(not_keep),
                "independent_studies": not_keep["doi"].str.lower().nunique(),
                "valid_age_rows": int(not_keep["n_valid_ages"].sum()),
                "note": "未删除原始记录，也未改原始标签；此处仅是衍生训练清单的地质裁定。",
            },
        ]
    )
    comparison.to_csv(DERIVED / "E1_comparison.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_zip": str(SRC_ZIP),
        "source_zip_sha256": sha256(SRC_ZIP),
        "original_e1_metadata_rows": len(full_e1),
        "original_e1_role_counts": full_e1["dataset_role"].value_counts(dropna=False).to_dict(),
        "original_candidate_samples": len(candidates),
        "original_candidate_studies": candidates["doi"].str.lower().nunique(),
        "original_candidate_grains": int(candidates["n_valid_ages"].sum()),
        "retained_samples": len(keep),
        "retained_studies": keep["doi"].str.lower().nunique(),
        "retained_grains_valid_age": int(keep["n_valid_ages"].sum()),
        "retained_grains_with_breitfeld_author_accept_only": int(keep["n_valid_ages"].sum()) - 1025 + 837,
        "external_or_deferred_samples": len(not_keep),
        "external_or_deferred_grains": int(not_keep["n_valid_ages"].sum()),
        "original_candidate_exact_duplicate_spectra": int(candidates["spectrum_fingerprint"].duplicated().sum()),
        "original_candidate_duplicate_sample_uids": int(candidates["sample_instance_uid"].duplicated().sum()),
        "original_candidate_exact_duplicates_elsewhere_in_current_db": len(all_fingerprint_matches) - len(candidates),
        "original_candidate_uid_collisions_elsewhere_in_current_db": len(all_uid_matches) - len(candidates),
        "river_source_physical_ids": len(source_headers),
        "river_source_analyses": int(author_flag.isin(["Y", "N"]).sum()),
        "river_source_accepted": int(author_flag.eq("Y").sum()),
        "retained_sample_ids": keep["sample_id"].tolist(),
        "excluded_or_deferred_sample_ids": not_keep["sample_id"].tolist(),
    }
    (DERIVED / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = [
        "sample_id", "sample_instance_uid", "study_id", "doi", "material_group",
        "formation", "formation_audit", "latitude", "longitude", "n_valid_ages",
        "n_author_accepted_known", "source_component", "audit_decision",
        "audit_reason", "source_identity_note", "source_evidence_url",
    ]
    def records(frame: pd.DataFrame, columns: list[str]) -> list[list[object]]:
        matrix = frame[columns].where(pd.notna(frame[columns]), None).values.tolist()
        return [[None if pd.isna(v) else v for v in row] for row in matrix]

    payload = {
        "summary": summary,
        "comparison": {"columns": comparison.columns.tolist(), "rows": records(comparison, comparison.columns.tolist())},
        "retained": {"columns": selected, "rows": records(keep, selected)},
        "external": {"columns": selected, "rows": records(not_keep, selected)},
        "studies": {"columns": study.columns.tolist(), "rows": records(study, study.columns.tolist())},
        "river_source": {"columns": source_headers.columns.tolist(), "rows": records(source_headers, source_headers.columns.tolist())},
    }
    (DERIVED / "workbook_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
