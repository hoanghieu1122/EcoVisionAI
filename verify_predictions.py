"""Kiểm tra smoke test model thật trên sáu ảnh mẫu."""

from utils.model_loader import load_metadata, load_model
from utils.paths import SAMPLE_IMAGES_DIR
from utils.prediction import predict_waste


def main() -> None:
    metadata = load_metadata()
    model = load_model()
    print(f"Input: {model.input_shape} | Output: {model.output_shape}")
    print("Nhãn:", list(metadata.class_names))
    for expected_label in metadata.class_names:
        image_path = SAMPLE_IMAGES_DIR / f"sample_{expected_label}.jpg"
        result = predict_waste(image_path, model=model, metadata=metadata)
        print(
            f"{image_path.name}: dự đoán={result.label}, "
            f"xác suất={result.confidence:.2%}, thời gian={result.processing_time_ms:.1f} ms"
        )


if __name__ == "__main__":
    main()
