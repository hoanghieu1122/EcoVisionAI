"""Đường dẫn tuyệt đối của dự án, không phụ thuộc thư mục chạy lệnh."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "waste_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

EVALUATION_DIR = BASE_DIR / "evaluation_results"
METRICS_PATH = EVALUATION_DIR / "test_evaluation_metrics.json"
CONFUSION_MATRIX_PATH = EVALUATION_DIR / "confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = EVALUATION_DIR / "classification_report.txt"
TRAINING_HISTORY_PATH = EVALUATION_DIR / "training_history.csv"
CONFUSION_MATRIX_IMAGE_PATH = EVALUATION_DIR / "confusion_matrix.png"
TRAINING_CHART_IMAGE_PATH = EVALUATION_DIR / "training_accuracy_loss.png"
SAMPLE_VERIFICATION_IMAGE_PATH = EVALUATION_DIR / "sample_prediction_verification.png"

SAMPLE_IMAGES_DIR = BASE_DIR / "sample_test_images"
ASSETS_DIR = BASE_DIR / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "history.db"
