#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp1_training_utils import (
    ANOMALY_DEFAULT,
    NORMAL_DEFAULT,
    evaluate_predictions,
    extract_feature_frame,
    load_exp1_dataset,
    measure_inference_latency,
    save_json,
    save_pickle,
    train_test_split_exp1,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_models" / "svm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SVM on exp1 handover windows.")
    parser.add_argument("--normal-csv", type=Path, default=NORMAL_DEFAULT)
    parser.add_argument("--anomaly-csv", type=Path, default=ANOMALY_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_exp1_dataset(args.normal_csv.expanduser().resolve(), args.anomaly_csv.expanduser().resolve())
    feature_frame = extract_feature_frame(frame)
    X_train, X_test, y_train, y_test = train_test_split_exp1(feature_frame, frame["label"], args.test_size, args.seed)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    def _predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = model.predict_proba(X)[:, 1]
        preds = model.predict(X).astype(int)
        return preds, probabilities

    y_pred, y_score, inference_latency_ms = measure_inference_latency(_predict, X_test)
    metrics = evaluate_predictions(y_test, y_pred, y_score)
    metrics.update(
        {
            "inference_latency_ms": inference_latency_ms,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "feature_count": int(feature_frame.shape[1]),
        }
    )

    save_pickle(output_dir / "svm.pkl", {"model": model})
    save_json(
        output_dir / "metrics.json",
        {
            "model": "SVM",
            "metrics": metrics,
            "features": list(feature_frame.columns),
            "rows": int(len(frame)),
        },
    )
    save_json(
        output_dir / "catalog_entry.json",
        {
            "name": "SVM",
            "task_type": "binary classification",
            "input_type": "handover-window KPI features",
            "accuracy": metrics["f1"],
            "inference_latency_ms": metrics["inference_latency_ms"],
            "training_cost": "medium",
            "label_requirement": "labeled",
            "tags": ["exp1", "handover", "tabular", "supervised", "margin-based"],
            "description": "SVM trained on normal vs anomaly handover windows.",
        },
    )

    print("=" * 72)
    print("SVM training completed")
    print(f"Output folder: {output_dir}")
    print(f"Feature count: {feature_frame.shape[1]}")
    print(metrics)


if __name__ == "__main__":
    main()
