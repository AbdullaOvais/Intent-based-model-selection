#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean


CELL_COLUMNS = ["time_s", "imsi", "cell_id", "rnti"]
HANDOVER_COLUMNS = ["time_s", "imsi", "rnti", "target_cell_id"]

PDCP_COLUMNS = [
    "start_s",
    "end_s",
    "cell_id",
    "imsi",
    "rnti",
    "lcid",
    "n_tx_pdus",
    "tx_bytes",
    "n_rx_pdus",
    "rx_bytes",
    "delay_s",
    "delay_std_dev_s",
    "delay_min_s",
    "delay_max_s",
    "pdu_size_bytes",
    "pdu_size_std_dev_bytes",
    "pdu_size_min_bytes",
    "pdu_size_max_bytes",
]

CELL_INT_COLUMNS = {"imsi", "cell_id", "rnti"}
HANDOVER_INT_COLUMNS = {"imsi", "rnti", "target_cell_id"}
PDCP_INT_COLUMNS = {
    "cell_id",
    "imsi",
    "rnti",
    "lcid",
    "n_tx_pdus",
    "tx_bytes",
    "n_rx_pdus",
    "rx_bytes",
    "pdu_size_bytes",
}
PDCP_FLOAT_COLUMNS = {
    "start_s",
    "end_s",
    "delay_s",
    "delay_std_dev_s",
    "delay_min_s",
    "delay_max_s",
    "pdu_size_std_dev_bytes",
    "pdu_size_min_bytes",
    "pdu_size_max_bytes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ns-O-RAN trace logs to CSV and extract KPI summaries."
    )
    parser.add_argument("--cell-id-stats", type=Path, default=None)
    parser.add_argument("--handover-stats", type=Path, default=None)
    parser.add_argument("--pdcp-stats", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _split_ws(line: str) -> list[str]:
    return re.split(r"\s+", line.strip())


def _read_table(path: Path, columns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("%") or line.startswith("#"):
            continue
        parts = _split_ws(line)
        if len(parts) != len(columns):
            raise ValueError(
                f"{path}: expected {len(columns)} columns, got {len(parts)} in line:\n{line}"
            )
        rows.append(dict(zip(columns, parts)))
    return rows


def _as_int(value: str) -> int:
    return int(float(value))


def _as_float(value: str) -> float:
    return float(value)


def parse_cell_stats(path: Path) -> list[dict[str, object]]:
    raw_rows = _read_table(path, CELL_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        rows.append(
            {
                "time_s": _as_float(row["time_s"]),
                "imsi": _as_int(row["imsi"]),
                "cell_id": _as_int(row["cell_id"]),
                "rnti": _as_int(row["rnti"]),
            }
        )
    return rows


def parse_handover_stats(path: Path) -> list[dict[str, object]]:
    raw_rows = _read_table(path, HANDOER_COLUMNS := HANDOER_COLUMNS if False else HANDOVER_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        rows.append(
            {
                "time_s": _as_float(row["time_s"]),
                "imsi": _as_int(row["imsi"]),
                "rnti": _as_int(row["rnti"]),
                "target_cell_id": _as_int(row["target_cell_id"]),
            }
        )
    return rows


def parse_pdcp_stats(path: Path) -> list[dict[str, object]]:
    raw_rows = _read_table(path, PDCP_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in raw_rows:
        typed: dict[str, object] = {}
        for col in PDCP_COLUMNS:
            if col in PDCP_INT_COLUMNS:
                typed[col] = _as_int(row[col])
            elif col in PDCP_FLOAT_COLUMNS:
                typed[col] = _as_float(row[col])
            else:
                typed[col] = row[col]
        rows.append(typed)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cell_transitions(cell_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    transitions: list[dict[str, object]] = []
    rows = sorted(cell_rows, key=lambda r: (int(r["imsi"]), float(r["time_s"])))

    last_by_ue: dict[int, dict[str, object]] = {}
    for row in rows:
        imsi = int(row["imsi"])
        prev = last_by_ue.get(imsi)

        if prev is not None and int(prev["cell_id"]) != int(row["cell_id"]):
            transitions.append(
                {
                    "imsi": imsi,
                    "time_s": float(row["time_s"]),
                    "from_cell_id": int(prev["cell_id"]),
                    "to_cell_id": int(row["cell_id"]),
                    "prev_time_s": float(prev["time_s"]),
                    "delta_s": float(row["time_s"]) - float(prev["time_s"]),
                    "from_rnti": int(prev["rnti"]),
                    "to_rnti": int(row["rnti"]),
                }
            )

        last_by_ue[imsi] = row

    return transitions


def build_summary(
    cell_rows: list[dict[str, object]] | None,
    handover_rows: list[dict[str, object]] | None,
    pdcp_rows: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    summary: dict[str, object] = {}

    if cell_rows:
        imsis = {int(r["imsi"]) for r in cell_rows}
        cells = {int(r["cell_id"]) for r in cell_rows}
        transitions = build_cell_transitions(cell_rows)

        summary["cell_rows"] = len(cell_rows)
        summary["unique_ues"] = len(imsis)
        summary["unique_cells"] = len(cells)
        summary["cell_transition_changes"] = len(transitions)

    if handover_rows:
        ues = {int(r["imsi"]) for r in handover_rows}
        times = [float(r["time_s"]) for r in handover_rows]

        summary["handover_events"] = len(handover_rows)
        summary["handover_participant_ues"] = len(ues)
        summary["first_handover_time_s"] = min(times)
        summary["last_handover_time_s"] = max(times)

    if pdcp_rows:
        total_tx_pdus = sum(int(r["n_tx_pdus"]) for r in pdcp_rows)
        total_rx_pdus = sum(int(r["n_rx_pdus"]) for r in pdcp_rows)
        total_tx_bytes = sum(int(r["tx_bytes"]) for r in pdcp_rows)
        total_rx_bytes = sum(int(r["rx_bytes"]) for r in pdcp_rows)

        start_s = min(float(r["start_s"]) for r in pdcp_rows)
        end_s = max(float(r["end_s"]) for r in pdcp_rows)
        duration_s = max(end_s - start_s, 1e-9)

        delays = [float(r["delay_s"]) for r in pdcp_rows]
        rx_weights = [int(r["n_rx_pdus"]) for r in pdcp_rows]
        weighted_delay_den = sum(rx_weights)
        weighted_delay_s = (
            sum(float(r["delay_s"]) * int(r["n_rx_pdus"]) for r in pdcp_rows) / weighted_delay_den
            if weighted_delay_den > 0
            else mean(delays)
        )

        summary["pdcp_rows"] = len(pdcp_rows)
        summary["total_tx_pdus"] = total_tx_pdus
        summary["total_rx_pdus"] = total_rx_pdus
        summary["total_tx_bytes"] = total_tx_bytes
        summary["total_rx_bytes"] = total_rx_bytes
        summary["packet_loss_rate_pct"] = (
            ((total_tx_pdus - total_rx_pdus) / total_tx_pdus) * 100.0 if total_tx_pdus > 0 else 0.0
        )
        summary["approx_tx_throughput_mbps"] = (total_tx_bytes * 8.0 / duration_s) / 1_000_000.0
        summary["approx_rx_throughput_mbps"] = (total_rx_bytes * 8.0 / duration_s) / 1_000_000.0
        summary["mean_delay_s"] = mean(delays)
        summary["weighted_delay_s"] = weighted_delay_s
        summary["duration_s"] = duration_s

    return [summary] if summary else []


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_rows = parse_cell_stats(args.cell_id_stats) if args.cell_id_stats else None
    handover_rows = parse_handover_stats(args.handover_stats) if args.handover_stats else None
    pdcp_rows = parse_pdcp_stats(args.pdcp_stats) if args.pdcp_stats else None

    if cell_rows is not None:
        write_csv(out_dir / "CellIdStats.csv", cell_rows, CELL_COLUMNS)
        transitions = build_cell_transitions(cell_rows)
        write_csv(
            out_dir / "CellTransitions.csv",
            transitions,
            [
                "imsi",
                "time_s",
                "from_cell_id",
                "to_cell_id",
                "prev_time_s",
                "delta_s",
                "from_rnti",
                "to_rnti",
            ],
        )

    if handover_rows is not None:
        write_csv(out_dir / "CellIdStatsHandover.csv", handover_rows, HANDOVER_COLUMNS)


    if pdcp_rows is not None:
        write_csv(out_dir / "DlE2PdcpStats.csv", pdcp_rows, PDCP_COLUMNS)

    summary_rows = build_summary(cell_rows, handover_rows, pdcp_rows)
    if summary_rows:
        summary_fields = list(summary_rows[0].keys())
        write_csv(out_dir / "KpiSummary.csv", summary_rows, summary_fields)

        print("Saved:")
        for name in [
            "CellIdStats.csv",
            "CellIdStatsHandover.csv",
            "DlE2PdcpStats.csv",
            "CellTransitions.csv",
            "KpiSummary.csv",
        ]:
            path = out_dir / name
            if path.exists():
                print(f"  - {path}")


if __name__ == "__main__":
    main()
