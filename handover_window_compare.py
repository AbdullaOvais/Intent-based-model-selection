#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def read_handover_times(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 4:
            continue
        rows.append(
            {
                "handover_time_s": float(parts[0]),
                "imsi": int(float(parts[1])),
                "rnti": int(float(parts[2])),
                "target_cell_id": int(float(parts[3])),
            }
        )
    return pd.DataFrame(rows)


def summarize_subset(sub: pd.DataFrame, window_s: float) -> dict[str, float]:
    if sub.empty:
        return {
            "rows": 0,
            "tx_pdus": 0.0,
            "rx_pdus": 0.0,
            "tx_bytes": 0.0,
            "rx_bytes": 0.0,
            "loss_pct": 0.0,
            "thr_rx_mbps": 0.0,
            "thr_tx_mbps": 0.0,
            "mean_delay_s": 0.0,
            "weighted_delay_s": 0.0,
        }

    tx_pdus = float(sub["n_tx_pdus"].sum())
    rx_pdus = float(sub["n_rx_pdus"].sum())
    tx_bytes = float(sub["tx_bytes"].sum())
    rx_bytes = float(sub["rx_bytes"].sum())
    loss_pct = ((tx_pdus - rx_pdus) / tx_pdus * 100.0) if tx_pdus > 0 else 0.0
    thr_rx_mbps = rx_bytes * 8.0 / window_s / 1_000_000.0
    thr_tx_mbps = tx_bytes * 8.0 / window_s / 1_000_000.0
    mean_delay_s = float(sub["delay_s"].mean())
    weighted_delay_s = (
        float((sub["delay_s"] * sub["n_rx_pdus"]).sum()) / float(sub["n_rx_pdus"].sum())
        if float(sub["n_rx_pdus"].sum()) > 0
        else mean_delay_s
    )

    return {
        "rows": int(len(sub)),
        "tx_pdus": tx_pdus,
        "rx_pdus": rx_pdus,
        "tx_bytes": tx_bytes,
        "rx_bytes": rx_bytes,
        "loss_pct": loss_pct,
        "thr_rx_mbps": thr_rx_mbps,
        "thr_tx_mbps": thr_tx_mbps,
        "mean_delay_s": mean_delay_s,
        "weighted_delay_s": weighted_delay_s,
    }


def event_windows(pdcp: pd.DataFrame, handovers: pd.DataFrame, window_s: float) -> pd.DataFrame:
    mid = (pdcp["start_s"] + pdcp["end_s"]) / 2.0
    out = []

    for idx, row in handovers.reset_index(drop=True).iterrows():
        t = float(row["handover_time_s"])
        imsi = int(row["imsi"])
        target = int(row["target_cell_id"])

        pre = pdcp[(mid >= t - window_s) & (mid < t)]
        post = pdcp[(mid >= t) & (mid < t + window_s)]

        pre_m = summarize_subset(pre, window_s)
        post_m = summarize_subset(post, window_s)

        out.append(
            {
                "event_id": idx + 1,
                "handover_time_s": t,
                "imsi": imsi,
                "target_cell_id": target,
                "pre_rows": pre_m["rows"],
                "post_rows": post_m["rows"],
                "pre_loss_pct": pre_m["loss_pct"],
                "post_loss_pct": post_m["loss_pct"],
                "delta_loss_pct": post_m["loss_pct"] - pre_m["loss_pct"],
                "pre_thr_rx_mbps": pre_m["thr_rx_mbps"],
                "post_thr_rx_mbps": post_m["thr_rx_mbps"],
                "delta_thr_rx_mbps": post_m["thr_rx_mbps"] - pre_m["thr_rx_mbps"],
                "pre_mean_delay_s": pre_m["mean_delay_s"],
                "post_mean_delay_s": post_m["mean_delay_s"],
                "delta_mean_delay_s": post_m["mean_delay_s"] - pre_m["mean_delay_s"],
                "pre_weighted_delay_s": pre_m["weighted_delay_s"],
                "post_weighted_delay_s": post_m["weighted_delay_s"],
                "delta_weighted_delay_s": post_m["weighted_delay_s"] - pre_m["weighted_delay_s"],
            }
        )

    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Window-based KPI comparison around handover events.")
    parser.add_argument("--normal-pdcp", required=True)
    parser.add_argument("--normal-handover", required=True)
    parser.add_argument("--anomaly-pdcp", required=True)
    parser.add_argument("--anomaly-handover", required=True)
    parser.add_argument("--window", type=float, default=0.2, help="Half-window size in seconds")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal_pdcp = pd.read_csv(args.normal_pdcp)
    anomaly_pdcp = pd.read_csv(args.anomaly_pdcp)
    normal_ho = read_handover_times(Path(args.normal_handover))
    anomaly_ho = read_handover_times(Path(args.anomaly_handover))

    normal_events = event_windows(normal_pdcp, normal_ho, args.window)
    anomaly_events = event_windows(anomaly_pdcp, anomaly_ho, args.window)

    normal_events.to_csv(out_dir / "normal_handover_windows.csv", index=False)
    anomaly_events.to_csv(out_dir / "anomaly_handover_windows.csv", index=False)

    summary_rows = []
    for metric in ["loss_pct", "thr_rx_mbps", "mean_delay_s", "weighted_delay_s"]:
        summary_rows.append(
            {
                "metric": metric,
                "normal_pre_mean": normal_events[f"pre_{metric}"].mean(),
                "normal_post_mean": normal_events[f"post_{metric}"].mean(),
                "normal_delta_mean": normal_events[f"delta_{metric}"].mean(),
                "anomaly_pre_mean": anomaly_events[f"pre_{metric}"].mean(),
                "anomaly_post_mean": anomaly_events[f"post_{metric}"].mean(),
                "anomaly_delta_mean": anomaly_events[f"delta_{metric}"].mean(),
                "anomaly_minus_normal_post": anomaly_events[f"post_{metric}"].mean() - normal_events[f"post_{metric}"].mean(),
                "anomaly_minus_normal_delta": anomaly_events[f"delta_{metric}"].mean() - normal_events[f"delta_{metric}"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "handover_window_summary.csv", index=False)

    print(f"Saved: {out_dir / 'normal_handover_windows.csv'}")
    print(f"Saved: {out_dir / 'anomaly_handover_windows.csv'}")
    print(f"Saved: {out_dir / 'handover_window_summary.csv'}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
