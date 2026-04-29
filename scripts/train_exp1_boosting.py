#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from exp1_training_utils import (
    ANOMALY_DEFAULT,
    NORMAL_DEFAULT,
    evaluate_predictions,
    extract_feature_frame,
    load_exp1_dataset,
    measure_inference_latency,
    model_slug,
    preprocess_for_tree_model,
    save_json,
    save_pickle,
    train_test_split_exp1,
)

try:  # Optional: use XGBoost if installed.
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[assignment]


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_models" / "boosting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost or Gradient Boosting on exp1 handover windows.")
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
    X_train_imp, X_test_imp, _ = preprocess_for_tree_model(X_train, X_test)

    if XGBClassifier is not None:
        model_name = "XGBoost"
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        model_name = "Gradient Boosting"
        model = GradientBoostingClassifier(
            random_state=args.seed,
            n_estimators=250,
            learning_rate=0.05,
            subsample=0.9,
        )

    model.fit(X_train_imp, y_train)

    def _predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            raw = model.decision_function(X)
            probabilities = 1.0 / (1.0 + np.exp(-raw))
        preds = model.predict(X).astype(int)
        return preds, np.asarray(probabilities, dtype=np.float64)

    y_pred, y_score, inference_latency_ms = measure_inference_latency(_predict, X_test_imp)
    metrics = evaluate_predictions(y_test, y_pred, y_score)
    metrics.update(
        {
            "inference_latency_ms": inference_latency_ms,
            "train_samples": int(len(X_train_imp)),
            "test_samples": int(len(X_test_imp)),
            "feature_count": int(feature_frame.shape[1]),
            "model_variant": model_name,
        }
    )

    save_pickle(output_dir / f"{model_slug(model_name)}.pkl", {"model": model})
    save_json(
        output_dir / "metrics.json",
        {
            "model": model_name,
            "metrics": metrics,
            "features": list(feature_frame.columns),
            "rows": int(len(frame)),
        },
    )
    save_json(
        output_dir / "catalog_entry.json",
        {
            "name": model_name,
            "task_type": "binary classification",
            "input_type": "handover-window KPI features",
            "accuracy": metrics["f1"],
            "inference_latency_ms": metrics["inference_latency_ms"],
            "training_cost": "medium",
            "label_requirement": "labeled",
            "tags": ["exp1", "handover", "tabular", "supervised", "boosting"],
            "description": f"{model_name} trained on normal vs anomaly handover windows.",
        },
    )

    print("=" * 72)
    print(f"{model_name} training completed")
    print(f"Output folder: {output_dir}")
    print(f"Feature count: {feature_frame.shape[1]}")
    print(metrics)


if __name__ == "__main__":
    main()

