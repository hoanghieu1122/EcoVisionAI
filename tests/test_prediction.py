from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from utils.model_loader import load_metadata, load_model
from utils.paths import MODEL_PATH, SAMPLE_IMAGES_DIR
from utils.prediction import ImageProcessingError, PredictionError, predict_waste, preprocess_image
from utils.waste_information import CLASS_ORDER


def test_preprocess_shape_dtype_and_range():
    image = Image.new("RGBA", (31, 57), (255, 0, 127, 100))
    batch = preprocess_image(image)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= -1.0
    assert batch.max() <= 1.0
    assert np.isclose(batch[0, 0, 0, 0], 1.0)


def test_builtin_preprocess_flag_avoids_second_scaling():
    metadata = replace(load_metadata(), has_builtin_preprocess=True)
    image = Image.new("RGB", (10, 10), (255, 127, 0))
    batch = preprocess_image(image, metadata)
    assert batch[0, 0, 0].tolist() == [255.0, 127.0, 0.0]


def test_invalid_image_is_rejected():
    with pytest.raises(ImageProcessingError, match="Không thể đọc ảnh"):
        preprocess_image(b"khong-phai-anh")


def test_missing_model_is_reported(tmp_path):
    load_model.clear()
    from utils.model_loader import ModelLoadError

    with pytest.raises(ModelLoadError, match="Không tìm thấy"):
        load_model(str(tmp_path / "missing.keras"))


def test_real_model_loads_with_six_outputs():
    assert MODEL_PATH.is_file()
    model = load_model()
    assert model.output_shape[-1] == 6
    assert tuple(model.input_shape[-3:]) == (224, 224, 3)


@pytest.mark.parametrize("label", CLASS_ORDER)
def test_real_model_predicts_each_sample_without_fake_expectations(label):
    sample_path = SAMPLE_IMAGES_DIR / f"sample_{label}.jpg"
    assert sample_path.is_file()
    result = predict_waste(sample_path)
    assert result.label in CLASS_ORDER
    assert result.name_vi
    assert 0.0 <= result.confidence <= 1.0
    assert list(result.probabilities) == CLASS_ORDER
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert result.processing_time_ms >= 0


def test_non_probability_output_is_rejected():
    class FakeModel:
        def predict(self, _batch, verbose=0):
            return np.array([[2.0, 1.0, 0.0, -1.0, 0.5, 0.3]], dtype=np.float32)

    with pytest.raises(PredictionError, match="không phải phân phối xác suất"):
        predict_waste(Image.new("RGB", (224, 224)), model=FakeModel())
