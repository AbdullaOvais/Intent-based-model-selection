from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


NORMAL_DEFAULT = Path.home() / "Desktop" / "normal_handover_windows.csv"
ANOMALY_DEFAULT = Path.home() / "Desktop" / "anomaly_handover_windows.csv"


ID_COLUMNS = {
    "event_id",
    "handover_time_s",
    "imsi",
    "target_cell_id",
    "scenario",
    "label",
}

FEATURE_COLUMNS = [
    "pre_loss_pct",
    "post_loss_pct",
    "delta_loss_pct",
    "pre_thr_rx_mbps",
    "post_thr_rx_mbps",
    "delta_thr_rx_mbps",
    "pre_mean_delay_s",
    "post_mean_delay_s",
    "delta_mean_delay_s",
    "pre_weighted_delay_s",
    "post_weighted_delay_s",
    "delta_weighted_delay_s",
]


def load_exp1_dataset(normal_csv: Path, anomaly_csv: Path) -> pd.DataFrame:
    normal = read_window_file(normal_csv, label=0, scenario="normal")
    anomaly = read_window_file(anomaly_csv, label=1, scenario="anomaly")
    combined = pd.concat([normal, anomaly], ignore_index=True)
    if combined.empty:
        raise ValueError("The combined dataset is empty.")
    return combined


def read_window_file(path: Path, label: int, scenario: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty file: {path}")
    frame = frame.copy()
    frame["label"] = int(label)
    frame["scenario"] = scenario
    return frame


def extract_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.replace([np.inf, -np.inf], np.nan)


def train_test_split_exp1(
    feature_frame: pd.DataFrame,
    labels: Iterable[int],
    test_size: float = 0.25,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = feature_frame.to_numpy(dtype=np.float32)
    y = np.asarray(list(labels), dtype=np.int64)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def preprocess_for_tree_model(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train).astype(np.float32)
    X_test_imputed = imputer.transform(X_test).astype(np.float32)
    return X_train_imputed, X_test_imputed, imputer


def preprocess_for_linear_model(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed.astype(np.float32), X_test_imputed.astype(np.float32), imputer


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
    return metrics


def measure_inference_latency(predict_fn, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    y_pred, y_score = predict_fn(X_test)
    elapsed = time.perf_counter() - start
    latency_ms = 1000.0 * elapsed / max(1, len(X_test))
    return y_pred, y_score, float(latency_ms)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_pickle(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def model_slug(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )

