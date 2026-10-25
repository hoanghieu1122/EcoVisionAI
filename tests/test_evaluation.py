import pytest

from utils.evaluation import load_evaluation_data
from utils.waste_information import CLASS_ORDER


def test_evaluation_artifacts_are_consistent():
    data = load_evaluation_data()
    metrics = data.metrics
    assert metrics["test_num_samples"] == 380
    assert metrics["accuracy"] == pytest.approx(0.8421052631578947)
    assert metrics["macro_f1_score"] == pytest.approx(0.8269881033505215)
    assert data.confusion_matrix.to_numpy().sum() == 380
    assert data.classification_report["Support"].sum() == 380
    assert data.confusion_matrix.index.tolist() == CLASS_ORDER
    assert data.training_history["epoch"].tolist() == list(range(1, 23))
    assert data.training_history[["accuracy", "loss", "val_accuracy", "val_loss"]].notna().all().all()
