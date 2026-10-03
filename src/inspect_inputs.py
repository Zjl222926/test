"""Fail-fast audit of immutable model-ready inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
TRAIN = PACKAGE / "data" / "training"
TARGET = PACKAGE / "data" / "target" / "TARGET_Borneo_detrital_zircon_MASTER_v2.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    index = pd.read_csv(TRAIN / "model_inputs" / "model_index.csv")
    kde = pd.read_csv(TRAIN / "model_inputs" / "X_KDE_bw20_grid10.csv")
    with (TRAIN / "model_inputs" / "age_arrays.jsonl").open(encoding="utf-8") as stream:
        ages = [json.loads(line) for line in stream]
    target = pd.read_csv(TARGET, low_memory=False)

    assert len(index) == len(kde) == len(ages) == 117
    assert kde.shape == (117, 461)
    assert index["global_sample_uid"].is_unique
    assert index["E_label"].value_counts().to_dict() == {"E2": 53, "E3": 34, "E1": 30}
    assert index["cv_group"].nunique() == 21
    assert set(index["test_fold"]) == {0, 1, 2, 3}
    assert index.groupby("cv_group")["test_fold"].nunique().max() == 1
    assert [row["global_sample_uid"] for row in ages] == index["global_sample_uid"].tolist()
    assert np.isfinite(kde.to_numpy(float)).all()
    assert len(target) == 14838
    assert target["sample_instance_uid"].nunique() == 140
    assert set(target["target_type"]) == {"TARGET_STRATIGRAPHIC"}

    report = {
        "training_samples": 117,
        "training_studies": 21,
        "class_counts": index["E_label"].value_counts().to_dict(),
        "kde_shape": list(kde.shape),
        "target_samples": int(target["sample_instance_uid"].nunique()),
        "target_grains": len(target),
        "sha256": {
            "model_index.csv": sha256(TRAIN / "model_inputs" / "model_index.csv"),
            "X_KDE_bw20_grid10.csv": sha256(TRAIN / "model_inputs" / "X_KDE_bw20_grid10.csv"),
            "age_arrays.jsonl": sha256(TRAIN / "model_inputs" / "age_arrays.jsonl"),
            TARGET.name: sha256(TARGET),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

