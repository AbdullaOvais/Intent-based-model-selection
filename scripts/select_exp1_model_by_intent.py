#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ns_oran_intent_selector.catalog import load_catalog
from ns_oran_intent_selector.selector import parse_intent, select_model


DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_hosinr" / "model_catalog.json"
DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "artifacts" / "exp1_hosinr" / "model_index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a model from the final exp catalog using operator intent.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.catalog.expanduser().resolve())
    intent = parse_intent(args.intent)
    result = select_model(intent, catalog)

    index = {}
    if args.index.exists():
        index = json.loads(args.index.read_text(encoding="utf-8"))

    artifact = index.get(result.selected.name, {})
    payload = {
        "intent": args.intent,
        "selected_model": result.selected.name,
        "selected_reason": result.reason,
        "selected_confidence": result.confidence,
        "provider": result.provider,
        "catalog_accuracy": result.selected.accuracy,
        "catalog_latency_ms": result.selected.inference_latency_ms,
        "artifact": artifact,
    }

    print("=" * 72)
    print(f"Intent: {args.intent}")
    print(f"Selected model: {result.selected.name}")
    print(f"Reason: {result.reason}")
    # print(f"Confidence: {result.confidence}")
    # print(f"Catalog accuracy: {result.selected.accuracy:.4f}")
    print(f"Catalog latency: {result.selected.inference_latency_ms:.4f} ms")
    if artifact:
        print(f"Model path: {artifact.get('model_path', '')}")

    if args.output_json:
        args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved selection JSON: {args.output_json}")


if __name__ == "__main__":
    main()
