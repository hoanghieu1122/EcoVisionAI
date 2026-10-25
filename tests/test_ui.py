from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from utils.evaluation import load_evaluation_data
from utils.prediction import PredictionResult
from utils.waste_information import CLASS_ORDER, WASTE_INFO
from views.recognition import _probability_chart
from views.statistics import _confusion_matrix_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_app_navigation_exposes_all_five_views_without_exceptions():
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.radio) == 1

    expected_pages = [
        "Trang chủ",
        "Nhận diện AI",
        "Hướng dẫn phân loại",
        "Lịch sử nhận diện",
        "Thống kê & đánh giá",
    ]
    assert len(app.radio[0].options) == len(expected_pages)

    for page_name in expected_pages:
        app.radio[0].set_value(page_name).run()
        assert not app.exception, page_name


def test_confusion_matrix_keeps_real_data_order_and_readable_annotations():
    data = load_evaluation_data()
    display_names = [WASTE_INFO[label]["name_vi"] for label in CLASS_ORDER]
    figure = _confusion_matrix_figure(data.confusion_matrix, display_names)

    assert np.array_equal(np.asarray(figure.data[0].z), data.confusion_matrix.to_numpy())
    assert list(figure.data[0].x) == display_names
    assert list(figure.data[0].y) == display_names
    assert figure.layout.xaxis.title.text == "Nhãn dự đoán"
    assert figure.layout.yaxis.title.text == "Nhãn thực tế"
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"

    annotations = list(figure.layout.annotations)
    assert len(annotations) == 36
    annotation_colors = {annotation.font.color for annotation in annotations}
    assert annotation_colors == {"#0F172A", "#FFFFFF"}


def test_probability_chart_keeps_all_six_real_classes():
    probabilities = {label: value for label, value in zip(CLASS_ORDER, [0.05, 0.1, 0.15, 0.2, 0.4, 0.1])}
    result = PredictionResult(
        label="plastic",
        name_vi="Nhựa",
        confidence=0.4,
        probabilities=probabilities,
        processing_time_ms=12.5,
    )
    figure = _probability_chart(result)

    plotted_categories = {name for trace in figure.data for name in trace.y}
    expected_categories = {WASTE_INFO[label]["name_vi"] for label in CLASS_ORDER}
    assert plotted_categories == expected_categories
    assert figure.layout.font.color == "#0F172A"
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_theme_config_locks_light_mode_and_declares_required_colors():
    config_text = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in config_text
    assert 'primaryColor = "#10B981"' in config_text
    assert 'backgroundColor = "#F8FAFC"' in config_text
    assert 'secondaryBackgroundColor = "#FFFFFF"' in config_text
    assert 'textColor = "#0F172A"' in config_text
    assert "[theme.light]" not in config_text
    assert "[theme.dark]" not in config_text


def test_evaluation_kpis_remain_unchanged():
    metrics = load_evaluation_data().metrics
    assert metrics["accuracy"] == pytest.approx(0.8421052631578947)
    assert metrics["macro_f1_score"] == pytest.approx(0.8269881033505215)
    assert metrics["weighted_f1_score"] == pytest.approx(0.8418322042537193)
    assert metrics["test_num_samples"] == 380
