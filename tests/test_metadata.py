import json

import pytest

from utils.model_loader import ModelConfigurationError, load_metadata
from utils.paths import CLASS_NAMES_PATH, MODEL_CONFIG_PATH
from utils.waste_information import CLASS_ORDER


def test_real_metadata_has_exact_label_order():
    metadata = load_metadata()
    assert list(metadata.class_names) == CLASS_ORDER
    assert metadata.image_size == (224, 224)
    assert metadata.has_builtin_preprocess is False


def test_config_declares_expected_preprocessing():
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["input_shape"] == [224, 224, 3]
    assert config["preprocessing"]["method"] == "tf.keras.applications.mobilenet_v2.preprocess_input"
    assert config["preprocessing"]["resize_interpolation"] == "bilinear"


def test_invalid_label_order_is_rejected(tmp_path):
    class_path = tmp_path / "classes.json"
    class_path.write_text(json.dumps(list(reversed(CLASS_ORDER))), encoding="utf-8")
    with pytest.raises(ModelConfigurationError, match="Thứ tự nhãn"):
        load_metadata(class_path, MODEL_CONFIG_PATH)


def test_model_and_json_files_exist():
    assert CLASS_NAMES_PATH.is_file()
    assert MODEL_CONFIG_PATH.is_file()
