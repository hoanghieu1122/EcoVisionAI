"""Đọc cấu hình và tải trực tiếp mô hình Keras đã huấn luyện."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from utils.paths import CLASS_NAMES_PATH, MODEL_CONFIG_PATH, MODEL_PATH
from utils.waste_information import CLASS_ORDER


class ModelConfigurationError(RuntimeError):
    """Metadata hoặc cấu trúc mô hình không hợp lệ."""


class ModelLoadError(RuntimeError):
    """Không thể tải mô hình thật từ tệp được cung cấp."""


@dataclass(frozen=True)
class ModelMetadata:
    class_names: tuple[str, ...]
    config: dict[str, Any]
    image_size: tuple[int, int]
    has_builtin_preprocess: bool


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ModelConfigurationError(f"Không tìm thấy tệp cấu hình: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Không thể đọc JSON hợp lệ tại {path}: {exc}") from exc


def load_metadata(
    class_names_path: Path = CLASS_NAMES_PATH,
    config_path: Path = MODEL_CONFIG_PATH,
) -> ModelMetadata:
    """Đọc và kiểm tra metadata trước khi cho phép dự đoán."""
    class_names = _read_json(class_names_path)
    config = _read_json(config_path)

    if not isinstance(class_names, list) or not all(isinstance(item, str) for item in class_names):
        raise ModelConfigurationError("class_names.json phải là danh sách chuỗi.")
    if class_names != CLASS_ORDER:
        raise ModelConfigurationError(
            "Thứ tự nhãn không đúng. Cần giữ nguyên: " + ", ".join(CLASS_ORDER)
        )
    if not isinstance(config, dict):
        raise ModelConfigurationError("model_config.json phải là một đối tượng JSON.")
    if config.get("class_names") != class_names:
        raise ModelConfigurationError("Danh sách nhãn trong hai tệp JSON không đồng nhất.")
    if config.get("num_classes") != len(class_names):
        raise ModelConfigurationError("num_classes không khớp số nhãn.")

    image_size = config.get("image_size")
    input_shape = config.get("input_shape")
    if image_size != [224, 224] or input_shape != [224, 224, 3]:
        raise ModelConfigurationError("Mô hình yêu cầu ảnh RGB 224 × 224 × 3.")
    if config.get("color_mode") != "RGB":
        raise ModelConfigurationError("color_mode phải là RGB.")

    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ModelConfigurationError("Thiếu cấu hình preprocessing.")
    expected_method = "tf.keras.applications.mobilenet_v2.preprocess_input"
    if preprocessing.get("method") != expected_method:
        raise ModelConfigurationError(
            f"Phương pháp tiền xử lý không được hỗ trợ: {preprocessing.get('method')!r}."
        )
    if preprocessing.get("resize_interpolation") != "bilinear":
        raise ModelConfigurationError("Nội suy resize phải là bilinear.")
    has_builtin = preprocessing.get("model_has_builtin_preprocess")
    if not isinstance(has_builtin, bool):
        raise ModelConfigurationError("model_has_builtin_preprocess phải là true hoặc false.")

    return ModelMetadata(
        class_names=tuple(class_names),
        config=config,
        image_size=(int(image_size[0]), int(image_size[1])),
        has_builtin_preprocess=has_builtin,
    )


def _shape_last_dimension(shape: Any) -> int | None:
    if isinstance(shape, list):
        if len(shape) != 1:
            return None
        shape = shape[0]
    try:
        return int(shape[-1]) if shape[-1] is not None else None
    except (TypeError, IndexError, ValueError):
        return None


@st.cache_resource(show_spinner="Đang tải mô hình AI đã huấn luyện...")
def load_model(model_path: str = str(MODEL_PATH)):
    """Tải model `.keras` một lần; không có mô hình thay thế hoặc fallback."""
    path = Path(model_path)
    if not path.is_file():
        raise ModelLoadError(f"Không tìm thấy mô hình đã huấn luyện: {path}")

    metadata = load_metadata()
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise ModelLoadError(
            "Không thể tải models/waste_model.keras. Hãy kiểm tra TensorFlow tương thích "
            f"(mô hình được xuất bằng TensorFlow 2.21.0). Chi tiết: {exc}"
        ) from exc

    output_classes = _shape_last_dimension(getattr(model, "output_shape", None))
    if output_classes != len(metadata.class_names):
        raise ModelLoadError(
            f"Đầu ra mô hình có {output_classes} lớp, nhưng metadata có {len(metadata.class_names)} lớp."
        )

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0] if len(input_shape) == 1 else None
    try:
        actual_input = tuple(int(value) for value in input_shape[-3:])
    except (TypeError, ValueError):
        actual_input = None
    if actual_input != (224, 224, 3):
        raise ModelLoadError(f"Đầu vào mô hình không đúng 224 × 224 × 3: {input_shape}")

    return model
