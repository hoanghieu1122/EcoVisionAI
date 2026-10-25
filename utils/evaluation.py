"""Đọc và kiểm tra dữ liệu đánh giá thật đã xuất từ Colab."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from utils.paths import CONFUSION_MATRIX_PATH, METRICS_PATH, TRAINING_HISTORY_PATH
from utils.waste_information import CLASS_ORDER


class EvaluationDataError(RuntimeError):
    """Dữ liệu đánh giá thiếu hoặc không nhất quán."""


@dataclass(frozen=True)
class EvaluationData:
    metrics: dict
    confusion_matrix: pd.DataFrame
    classification_report: pd.DataFrame
    training_history: pd.DataFrame


def _read_metrics() -> dict:
    if not METRICS_PATH.is_file():
        raise EvaluationDataError(f"Không tìm thấy {METRICS_PATH}")
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"Không đọc được dữ liệu đánh giá: {exc}") from exc
    required = {"test_num_samples", "accuracy", "macro_f1_score", "per_class_report", "class_names"}
    if not required.issubset(metrics):
        raise EvaluationDataError("test_evaluation_metrics.json thiếu trường bắt buộc.")
    if metrics["class_names"] != CLASS_ORDER:
        raise EvaluationDataError("Thứ tự nhãn trong dữ liệu đánh giá không đúng.")
    return metrics


def _read_confusion_matrix(metrics: dict) -> pd.DataFrame:
    if not CONFUSION_MATRIX_PATH.is_file():
        raise EvaluationDataError(f"Không tìm thấy {CONFUSION_MATRIX_PATH}")
    try:
        frame = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
    except Exception as exc:
        raise EvaluationDataError(f"Không đọc được confusion_matrix.csv: {exc}") from exc
    if frame.index.tolist() != CLASS_ORDER or frame.columns.tolist() != CLASS_ORDER:
        raise EvaluationDataError("Nhãn của confusion matrix không đúng thứ tự chuẩn.")
    if frame.shape != (6, 6):
        raise EvaluationDataError("Confusion matrix phải có kích thước 6 × 6.")
    if int(frame.to_numpy().sum()) != int(metrics["test_num_samples"]):
        raise EvaluationDataError("Tổng confusion matrix không khớp số mẫu kiểm thử.")
    return frame


def _classification_frame(metrics: dict) -> pd.DataFrame:
    report = metrics["per_class_report"]
    rows = []
    for label in CLASS_ORDER:
        if label not in report:
            raise EvaluationDataError(f"Classification report thiếu lớp {label}.")
        row = report[label]
        rows.append(
            {
                "Nhãn AI": label,
                "Precision": float(row["precision"]),
                "Recall": float(row["recall"]),
                "F1-score": float(row["f1-score"]),
                "Support": int(row["support"]),
            }
        )
    frame = pd.DataFrame(rows)
    if int(frame["Support"].sum()) != int(metrics["test_num_samples"]):
        raise EvaluationDataError("Tổng support không khớp số mẫu kiểm thử.")
    return frame


def _read_training_history() -> pd.DataFrame:
    if not TRAINING_HISTORY_PATH.is_file():
        raise EvaluationDataError(f"Không tìm thấy {TRAINING_HISTORY_PATH}")
    try:
        frame = pd.read_csv(TRAINING_HISTORY_PATH)
    except Exception as exc:
        raise EvaluationDataError(f"Không đọc được training_history.csv: {exc}") from exc
    required = {"epoch", "accuracy", "loss", "val_accuracy", "val_loss", "learning_rate"}
    if not required.issubset(frame.columns):
        raise EvaluationDataError("training_history.csv thiếu cột bắt buộc.")
    if frame.empty:
        raise EvaluationDataError("training_history.csv không có dữ liệu.")
    display = frame.copy()
    display["epoch"] = display["epoch"].astype(int) + 1
    return display


def load_evaluation_data() -> EvaluationData:
    """Tải toàn bộ dữ liệu đánh giá, không sinh dữ liệu thay thế."""
    metrics = _read_metrics()
    return EvaluationData(
        metrics=metrics,
        confusion_matrix=_read_confusion_matrix(metrics),
        classification_report=_classification_frame(metrics),
        training_history=_read_training_history(),
    )
