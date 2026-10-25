"""Tiền xử lý ảnh và dự đoán bằng mô hình thật."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from utils.model_loader import ModelMetadata, load_metadata, load_model
from utils.waste_information import vietnamese_name


class ImageProcessingError(ValueError):
    """Ảnh đầu vào không thể giải mã hoặc xử lý."""


class PredictionError(RuntimeError):
    """Đầu ra mô hình không hợp lệ hoặc dự đoán thất bại."""


@dataclass(frozen=True)
class PredictionResult:
    label: str
    name_vi: str
    confidence: float
    probabilities: dict[str, float]
    processing_time_ms: float


def confidence_threshold() -> float:
    """Đọc ngưỡng cảnh báo từ môi trường, mặc định 60%."""
    raw_value = os.getenv("ECOVISION_CONFIDENCE_THRESHOLD", "0.60")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise PredictionError("ECOVISION_CONFIDENCE_THRESHOLD phải là một số.") from exc
    if not 0.0 <= value <= 1.0:
        raise PredictionError("ECOVISION_CONFIDENCE_THRESHOLD phải nằm trong khoảng 0 đến 1.")
    return value


def open_image(image_input: Image.Image | str | Path | bytes | BinaryIO) -> Image.Image:
    """Mở ảnh từ Pillow, đường dẫn, bytes hoặc file-like và nạp dữ liệu ngay."""
    try:
        if isinstance(image_input, Image.Image):
            image = image_input.copy()
        elif isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        else:
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            image = Image.open(image_input)
        image.load()
        return image
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ImageProcessingError(
            "Không thể đọc ảnh. Vui lòng chọn tệp JPG, JPEG hoặc PNG hợp lệ."
        ) from exc


def preprocess_image(
    image_input: Image.Image | str | Path | bytes | BinaryIO,
    metadata: ModelMetadata | None = None,
) -> np.ndarray:
    """Tạo batch float32 theo đúng cấu hình MobileNetV2 của gói model."""
    metadata = metadata or load_metadata()
    image = open_image(image_input)
    resized = image.convert("RGB").resize(metadata.image_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    if not metadata.has_builtin_preprocess:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        batch = preprocess_input(batch.copy())

    expected_shape = (1, metadata.image_size[0], metadata.image_size[1], 3)
    if batch.shape != expected_shape or batch.dtype != np.float32:
        raise ImageProcessingError(
            f"Batch ảnh không đúng yêu cầu: shape={batch.shape}, dtype={batch.dtype}."
        )
    return batch


def _validate_probabilities(raw_output: np.ndarray, class_count: int) -> np.ndarray:
    output = np.asarray(raw_output, dtype=np.float64)
    if output.ndim == 2 and output.shape[0] == 1:
        output = output[0]
    if output.shape != (class_count,):
        raise PredictionError(
            f"Đầu ra mô hình phải có {class_count} xác suất, nhận được shape {output.shape}."
        )
    if not np.all(np.isfinite(output)):
        raise PredictionError("Đầu ra mô hình chứa giá trị không hữu hạn.")
    if np.any(output < -1e-6) or np.any(output > 1.0 + 1e-6):
        raise PredictionError("Đầu ra không phải phân phối xác suất trong khoảng [0, 1].")
    if not np.isclose(float(output.sum()), 1.0, atol=1e-3):
        raise PredictionError(
            "Tổng đầu ra mô hình không bằng 1. Không tự áp dụng softmax để tránh thay đổi mô hình."
        )
    return output.astype(np.float32)


def predict_waste(
    image_input: Image.Image | str | Path | bytes | BinaryIO,
    model=None,
    metadata: ModelMetadata | None = None,
) -> PredictionResult:
    """Chạy dự đoán thật và trả kết quả có đủ sáu xác suất."""
    metadata = metadata or load_metadata()
    model = model if model is not None else load_model()
    batch = preprocess_image(image_input, metadata)

    try:
        started = time.perf_counter()
        raw_output = model.predict(batch, verbose=0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        raise PredictionError(f"Mô hình không thể xử lý ảnh: {exc}") from exc

    probabilities_array = _validate_probabilities(raw_output, len(metadata.class_names))
    best_index = int(np.argmax(probabilities_array))
    label = metadata.class_names[best_index]
    probabilities = {
        class_name: float(probabilities_array[index])
        for index, class_name in enumerate(metadata.class_names)
    }
    return PredictionResult(
        label=label,
        name_vi=vietnamese_name(label),
        confidence=probabilities[label],
        probabilities=probabilities,
        processing_time_ms=elapsed_ms,
    )
