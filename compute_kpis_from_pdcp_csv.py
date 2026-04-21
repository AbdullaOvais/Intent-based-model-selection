#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "start_s", "end_s", "n_tx_pdus", "tx_bytes",
        "n_rx_pdus", "rx_bytes", "delay_s"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df.copy()

    duration_s = (out["end_s"] - out["start_s"]).astype(float)
    duration_s = duration_s.where(duration_s > 0, 1e-9)

    out["row_duration_s"] = duration_s

    out["throughput_tx_mbps"] = (out["tx_bytes"].astype(float) * 8.0) / duration_s / 1_000_000.0
    out["throughput_rx_mbps"] = (out["rx_bytes"].astype(float) * 8.0) / duration_s / 1_000_000.0

    out["packet_loss_pct"] = (
        (out["n_tx_pdus"].astype(float) - out["n_rx_pdus"].astype(float))
        / out["n_tx_pdus"].replace(0, pd.NA).astype(float)
        * 100.0
    ).fillna(0.0)

    out["delay_ms"] = out["delay_s"].astype(float) * 1000.0

    # Optional extra metric:
    # received bytes normalized by time, same as throughput_rx_mbps but kept for clarity.
    out["goodput_rx_mbps"] = out["throughput_rx_mbps"]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment DlE2PdcpStats.csv with row-level KPI columns")
    parser.add_argument("--input-csv", required=True, help="Path to DlE2PdcpStats.csv")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output path. If omitted, overwrites the input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv) if args.output_csv else input_path

    df = pd.read_csv(input_path)
    augmented = add_metrics(df)
    augmented.to_csv(output_path, index=False)

    print(f"Saved augmented CSV to: {output_path}")
    print(f"Rows: {len(augmented)}")
    print("Added columns: row_duration_s, throughput_tx_mbps, throughput_rx_mbps, packet_loss_pct, delay_ms, goodput_rx_mbps")


if __name__ == "__main__":
    main()
