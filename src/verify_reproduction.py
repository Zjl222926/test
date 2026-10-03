"""Compare regenerated paper outputs with the frozen reference outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
GENERATED_METHOD = PACKAGE / "outputs" / "method_comparison_cpu"
REFERENCE_METHOD = PACKAGE / "reference_outputs" / "method_comparison"
GENERATED_TARGET = PACKAGE / "outputs" / "target_rf"
REFERENCE_TARGET = PACKAGE / "reference_outputs" / "target_rf"


def compare_predictions(generated: Path, reference: Path, keys: list[str]) -> None:
    left, right = pd.read_csv(generated), pd.read_csv(reference)
    assert len(left) == len(right)
    for key in keys:
        assert left[key].astype(str).tolist() == right[key].astype(str).tolist(), key


def main() -> None:
    compare_predictions(
        GENERATED_METHOD / "main_oof_predictions.csv",
        REFERENCE_METHOD / "main_oof_predictions.csv",
        ["method", "fold", "global_sample_uid", "true_label", "predicted_label"],
    )
    generated_summary = pd.read_csv(GENERATED_METHOD / "main_method_comparison.csv")
    reference_summary = pd.read_csv(REFERENCE_METHOD / "main_method_comparison.csv")
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "E1_recall", "E2_recall", "E3_recall"]
    assert generated_summary.method.tolist() == reference_summary.method.tolist()
    assert np.allclose(generated_summary[metrics], reference_summary[metrics], atol=1e-12, rtol=0)

    compare_predictions(
        GENERATED_TARGET / "TARGET_predictions_key.csv",
        REFERENCE_TARGET / "TARGET_predictions_key.csv",
        ["sample_instance_uid", "RF_predicted_class", "Pearson_R2_predicted_class",
         "KS_predicted_class", "Wasserstein_1_predicted_class", "quality_flags"],
    )
    generated_target = pd.read_csv(GENERATED_TARGET / "TARGET_predictions_key.csv")
    reference_target = pd.read_csv(REFERENCE_TARGET / "TARGET_predictions_key.csv")
    probability_columns = ["RF_probability_E1", "RF_probability_E2", "RF_probability_E3",
                           "RF_max_probability_uncalibrated", "RF_top2_probability_margin",
                           "RF_normalized_entropy_uncertainty"]
    assert np.allclose(generated_target[probability_columns], reference_target[probability_columns], atol=1e-12, rtol=0)

    report = {
        "status": "PASS",
        "main_oof_predictions_identical": True,
        "main_metrics_identical_within_1e-12": True,
        "target_classes_and_flags_identical": True,
        "target_probabilities_identical_within_1e-12": True,
        "note": "TPOT is verified separately because its fixed time-budget search can vary across operating systems.",
    }
    out = PACKAGE / "outputs" / "reproduction_verification.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

