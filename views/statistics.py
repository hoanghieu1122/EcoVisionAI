"""Trang thống kê và đánh giá mô hình bằng dữ liệu thật."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.evaluation import EvaluationDataError, load_evaluation_data
from utils.paths import (
    CONFUSION_MATRIX_IMAGE_PATH,
    SAMPLE_VERIFICATION_IMAGE_PATH,
    TRAINING_CHART_IMAGE_PATH,
)
from utils.ui import apply_plotly_theme, emerald_scale, kpi_card, render_header, render_section
from utils.waste_information import CLASS_ORDER, WASTE_INFO


def _training_figure(history, metric: str, validation_metric: str, title: str, y_title: str):
    """Tạo biểu đồ quá trình huấn luyện với màu Emerald/Teal."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history[metric],
            mode="lines+markers",
            name="Huấn luyện",
            line={"color": "#059669", "width": 3},
            marker={"size": 6, "color": "#059669", "line": {"width": 0}},
            hovertemplate="Epoch %{x}<br>Huấn luyện: %{y:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history[validation_metric],
            mode="lines+markers",
            name="Validation",
            line={"color": "#0D9488", "width": 3, "dash": "dot"},
            marker={"size": 6, "color": "#0D9488", "line": {"width": 0}},
            hovertemplate="Epoch %{x}<br>Validation: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "font": {"color": "#064E3B", "size": 17}},
        xaxis_title="Epoch",
        yaxis_title=y_title,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#334155"},
        },
    )
    return apply_plotly_theme(
        figure,
        height=390,
        margin={"l": 58, "r": 24, "t": 78, "b": 52},
    )


def _confusion_matrix_figure(matrix, display_names: list[str]):
    """Tạo confusion matrix giữ nguyên dữ liệu/thứ tự và tự đổi màu chữ từng ô."""
    values = matrix.to_numpy(dtype=int)
    max_value = int(values.max()) if values.size else 0
    text_threshold = max_value * 0.5

    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=display_names,
            y=display_names,
            colorscale=list(emerald_scale()),
            zmin=0,
            zmax=max_value,
            xgap=2,
            ygap=2,
            hovertemplate="Nhãn thực tế: %{y}<br>Nhãn dự đoán: %{x}<br>Số mẫu: %{z}<extra></extra>",
            colorbar={
                "title": {"text": "Số mẫu", "font": {"color": "#334155", "size": 13}},
                "tickfont": {"color": "#334155", "size": 12},
                "outlinewidth": 0,
                "thickness": 16,
                "len": 0.82,
            },
        )
    )

    annotations = []
    for row_index, row_name in enumerate(display_names):
        for column_index, column_name in enumerate(display_names):
            value = int(values[row_index, column_index])
            annotations.append(
                {
                    "x": column_name,
                    "y": row_name,
                    "text": str(value),
                    "showarrow": False,
                    "font": {
                        "color": "#FFFFFF" if value > text_threshold else "#0F172A",
                        "size": 14,
                        "family": "Plus Jakarta Sans, Inter, sans-serif",
                    },
                }
            )

    figure.update_layout(
        annotations=annotations,
        xaxis={
            "title": "Nhãn dự đoán",
            "side": "bottom",
            "showgrid": False,
            "ticks": "",
            "tickangle": -18,
            "tickfont": {"color": "#334155", "size": 13},
            "constrain": "domain",
        },
        yaxis={
            "title": "Nhãn thực tế",
            "showgrid": False,
            "ticks": "",
            "tickfont": {"color": "#334155", "size": 13},
            "autorange": "reversed",
            "scaleanchor": None,
        },
    )
    figure = apply_plotly_theme(
        figure,
        height=600,
        margin={"l": 105, "r": 45, "t": 28, "b": 105},
    )
    figure.update_xaxes(showgrid=False, ticks="", tickangle=-18)
    figure.update_yaxes(showgrid=False, ticks="", autorange="reversed")
    return figure


def render() -> None:
    render_header(
        "Thống kê và đánh giá AI",
        "Kết quả thực nghiệm trên tập test độc lập, không phải thống kê sử dụng",
        "📊",
    )
    try:
        data = load_evaluation_data()
    except EvaluationDataError as exc:
        st.error(f"Không thể hiển thị đánh giá thật: {exc}", icon=":material/error:")
        st.info(
            "Ứng dụng không tạo dữ liệu thay thế khi artifact đánh giá bị thiếu hoặc sai.",
            icon=":material/verified_user:",
        )
        return

    metrics = data.metrics
    render_section("Các chỉ số cốt lõi", "Hiệu suất mô hình")
    metric_columns = st.columns(4)
    with metric_columns[0]:
        kpi_card("Test Accuracy", f"{metrics['accuracy']:.2%}", "✓", "380 ảnh test")
    with metric_columns[1]:
        kpi_card("Macro F1-Score", f"{metrics['macro_f1_score']:.4f}", "F1", "Trung bình 6 lớp")
    with metric_columns[2]:
        kpi_card(
            "Weighted F1",
            f"{metrics['weighted_f1_score']:.4f}",
            "W",
            "Có trọng số support",
        )
    with metric_columns[3]:
        kpi_card("Mẫu kiểm thử", f"{metrics['test_num_samples']:,}", "#", "Tập test độc lập")
    st.caption(
        "Các chỉ số được đọc trực tiếp từ evaluation_results/test_evaluation_metrics.json. "
        "Chúng mô tả tập test 380 ảnh, không phải cam kết cho mọi ảnh thực tế."
    )

    render_section("Ma trận nhầm lẫn", "Phân tích theo lớp")
    display_names = [WASTE_INFO[label]["name_vi"] for label in CLASS_ORDER]
    confusion_figure = _confusion_matrix_figure(data.confusion_matrix, display_names)
    st.plotly_chart(
        confusion_figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        alt="Ma trận nhầm lẫn sáu lớp trên tập test 380 ảnh",
    )

    render_section("Báo cáo phân loại", "Precision · Recall · F1")
    report = data.classification_report.copy()
    report["Tên tiếng Việt"] = report["Nhãn AI"].map(lambda label: WASTE_INFO[label]["name_vi"])
    report = report[["Nhãn AI", "Tên tiếng Việt", "Precision", "Recall", "F1-score", "Support"]]
    st.dataframe(
        report,
        width="stretch",
        hide_index=True,
        column_config={
            "Nhãn AI": st.column_config.TextColumn(width="small"),
            "Tên tiếng Việt": st.column_config.TextColumn(width="medium"),
            "Precision": st.column_config.NumberColumn(format="%.4f", width="small"),
            "Recall": st.column_config.NumberColumn(format="%.4f", width="small"),
            "F1-score": st.column_config.NumberColumn(format="%.4f", width="small"),
            "Support": st.column_config.NumberColumn(format="%d", width="small"),
        },
        alt="Báo cáo Precision Recall F1 và Support cho sáu lớp",
    )

    render_section("Quá trình huấn luyện", "22 epoch")
    accuracy_col, loss_col = st.columns(2)
    with accuracy_col:
        st.plotly_chart(
            _training_figure(
                data.training_history,
                "accuracy",
                "val_accuracy",
                "Accuracy theo epoch",
                "Accuracy",
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            alt="Accuracy huấn luyện và validation qua 22 epoch",
        )
    with loss_col:
        st.plotly_chart(
            _training_figure(
                data.training_history,
                "loss",
                "val_loss",
                "Loss theo epoch",
                "Loss",
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            alt="Loss huấn luyện và validation qua 22 epoch",
        )

    with st.expander("Xem biểu đồ gốc được xuất từ Colab", icon=":material/image:"):
        if CONFUSION_MATRIX_IMAGE_PATH.is_file():
            st.image(
                CONFUSION_MATRIX_IMAGE_PATH,
                caption="Confusion Matrix gốc",
                width="stretch",
                alt="Confusion matrix gốc xuất từ Colab",
            )
        if TRAINING_CHART_IMAGE_PATH.is_file():
            st.image(
                TRAINING_CHART_IMAGE_PATH,
                caption="Training Accuracy và Loss gốc",
                width="stretch",
                alt="Biểu đồ huấn luyện gốc xuất từ Colab",
            )
        if SAMPLE_VERIFICATION_IMAGE_PATH.is_file():
            st.image(
                SAMPLE_VERIFICATION_IMAGE_PATH,
                caption="Kiểm tra dự đoán mẫu trước khi đóng gói",
                width="stretch",
                alt="Ảnh kiểm tra dự đoán mẫu từ Colab",
            )
