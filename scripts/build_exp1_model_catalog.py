#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ns_oran_intent_selector.catalog import ModelCatalog, save_catalog
from ns_oran_intent_selector.schema import ModelSpec


DEFAULT_MODEL_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_hosinr" / "model_outputs"
DEFAULT_OUTPUT_CATALOG = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_hosinr" / "model_catalog.json"
DEFAULT_OUTPUT_SUMMARY = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_hosinr" / "model_results.csv"
DEFAULT_OUTPUT_INDEX = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_hosinr" / "model_index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final model catalog from exp1 model outputs.")
    parser.add_argument("--model-outputs-dir", type=Path, default=DEFAULT_MODEL_OUTPUT_DIR)
    parser.add_argument("--output-catalog", type=Path, default=DEFAULT_OUTPUT_CATALOG)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--output-index", type=Path, default=DEFAULT_OUTPUT_INDEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_outputs_dir = args.model_outputs_dir.expanduser().resolve()
    if not model_outputs_dir.exists():
        raise FileNotFoundError(f"Model outputs directory does not exist: {model_outputs_dir}")

    entries: list[ModelSpec] = []
    summary_rows: list[dict[str, object]] = []
    index_payload: dict[str, dict[str, str]] = {}

    for catalog_path in sorted(model_outputs_dir.rglob("catalog_entry.json")):
        model_dir = catalog_path.parent
        metric_path = model_dir / "metrics.json"
        entry = json.loads(catalog_path.read_text(encoding="utf-8"))
        metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else {}
        metric_values = metrics.get("metrics", metrics)

        name = str(entry.get("name", model_dir.name))
        accuracy = float(metric_values.get("accuracy", entry.get("accuracy", 0.0)))
        latency = float(metric_values.get("inference_latency_ms", entry.get("inference_latency_ms", 0.0)))
        description = (
            f"Trained on exp1 normal vs anomaly handover windows. "
            f"Test accuracy={accuracy:.3f}, F1={float(metric_values.get('f1', accuracy)):.3f}, latency={latency:.3f} ms."
        )

        spec = ModelSpec(
            name=name,
            task_type=str(entry.get("task_type", "binary classification")),
            input_type=str(entry.get("input_type", "handover-window KPI features")),
            accuracy=accuracy,
            inference_latency_ms=latency,
            training_cost=str(entry.get("training_cost", "medium")),
            label_requirement=str(entry.get("label_requirement", "labeled")),
            tags=tuple(entry.get("tags", ())),
            description=description,
        )
        entries.append(spec)

        summary_rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision": float(metric_values.get("precision", float("nan"))),
                "recall": float(metric_values.get("recall", float("nan"))),
                "f1": float(metric_values.get("f1", float("nan"))),
                "roc_auc": float(metric_values.get("roc_auc", float("nan"))),
                "inference_latency_ms": latency,
                "training_cost": spec.training_cost,
                "label_requirement": spec.label_requirement,
                "source_dir": str(model_dir),
            }
        )
        index_payload[name] = {
            "model_dir": str(model_dir),
            "model_path": str(next((p for p in model_dir.glob("*.pkl")), model_dir / f"{name}.pkl")),
            "metrics_path": str(metric_path) if metric_path.exists() else "",
            "catalog_entry_path": str(catalog_path),
        }

    if not entries:
        raise ValueError(f"No catalog_entry.json files found under: {model_outputs_dir}")

    catalog = ModelCatalog(entries)
    args.output_catalog.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_index.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    save_catalog(catalog, args.output_catalog)
    pd.DataFrame(summary_rows).sort_values(["f1", "accuracy", "inference_latency_ms"], ascending=[False, False, True]).to_csv(
        args.output_summary,
        index=False,
    )
    args.output_index.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Saved catalog: {args.output_catalog}")
    print(f"Saved summary: {args.output_summary}")
    print(f"Saved index:   {args.output_index}")
    print()
    for row in summary_rows:
        print(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, f1={row['f1']:.4f}, "
            f"latency={row['inference_latency_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()

