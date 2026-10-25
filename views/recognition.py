"""Trang nhận diện ảnh tải lên hoặc ảnh camera."""

from __future__ import annotations

import hashlib
import html
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.database import insert_prediction
from utils.model_loader import ModelConfigurationError, ModelLoadError, load_metadata, load_model
from utils.prediction import (
    ImageProcessingError,
    PredictionError,
    PredictionResult,
    confidence_threshold,
    open_image,
    predict_waste,
)
from utils.ui import apply_plotly_theme, render_header, render_section
from utils.waste_information import CLASS_ORDER, WASTE_INFO, get_waste_info


def _probability_chart(result: PredictionResult):
    """Tạo biểu đồ sáu xác suất với giao diện sáng và nhãn rõ ràng."""
    frame = pd.DataFrame(
        {
            "Nhãn": CLASS_ORDER,
            "Loại rác": [WASTE_INFO[label]["name_vi"] for label in CLASS_ORDER],
            "Xác suất": [result.probabilities[label] for label in CLASS_ORDER],
        }
    )
    colors = ["#10B981" if label == result.label else "#A7F3D0" for label in CLASS_ORDER]
    figure = px.bar(
        frame,
        x="Xác suất",
        y="Loại rác",
        orientation="h",
        text=frame["Xác suất"].map(lambda value: f"{value:.1%}"),
        color="Nhãn",
        color_discrete_map=dict(zip(CLASS_ORDER, colors)),
    )
    figure.update_traces(
        textposition="outside",
        textfont={"color": "#0F172A", "size": 12},
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    )
    figure.update_layout(
        xaxis={
            "title": "Xác suất",
            "tickformat": ".0%",
            "range": [0, max(1.0, float(frame["Xác suất"].max()) * 1.16)],
            "showgrid": True,
        },
        yaxis={
            "title": "",
            "categoryorder": "array",
            "categoryarray": list(reversed(frame["Loại rác"].tolist())),
            "showgrid": False,
        },
        bargap=0.32,
    )
    return apply_plotly_theme(
        figure,
        height=390,
        margin={"l": 12, "r": 64, "t": 22, "b": 42},
        show_legend=False,
    )


def _render_result(result: PredictionResult, image_name: str, source: str, source_hash: str) -> None:
    info = get_waste_info(result.label)
    left, right = st.columns([1, 1.35])
    with left:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Kết quả nhận diện</div>
              <div class="result-value">{html.escape(info['icon'])} {html.escape(result.name_vi)}</div>
              <div class="result-meta"><span>Nhãn AI</span><strong>{html.escape(result.label)}</strong></div>
              <div class="result-meta"><span>Độ tin cậy</span><strong>{result.confidence:.2%}</strong></div>
              <div class="result-meta"><span>Thời gian mô hình</span><strong>{result.processing_time_ms:.1f} ms</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        threshold = confidence_threshold()
        if result.confidence < threshold:
            st.warning(
                f"Mô hình chưa đủ chắc chắn (dưới {threshold:.0%}). Hãy thử ảnh rõ hơn, đủ sáng và chỉ có một vật thể chính.",
                icon=":material/warning:",
            )
        else:
            st.success(
                "Mô hình đã đưa ra lớp có xác suất cao nhất. Hãy kết hợp với kiểm tra thực tế.",
                icon=":material/check_circle:",
            )
        st.caption("Xác suất softmax không phải độ chính xác thực tế.")
    with right:
        st.plotly_chart(
            _probability_chart(result),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            alt="Phân bố xác suất dự đoán cho sáu nhóm rác",
        )

    render_section("Hướng dẫn phân loại và xử lý", "Khuyến nghị")
    with st.container(border=True):
        st.info(info["handling"], icon=":material/recycling:")
        st.write(f"**Khả năng tái chế:** {info['recyclability']}")

    st.markdown(
        "<div class='privacy-note'>🔒 Ứng dụng không lưu nội dung ảnh. Chỉ khi bạn đồng ý, metadata kết quả mới được ghi vào lịch sử.</div>",
        unsafe_allow_html=True,
    )
    consent_key = f"consent_{source_hash}"
    save_key = f"saved_{source_hash}"
    consent = st.checkbox("Tôi đồng ý lưu bản ghi kết quả này vào lịch sử", key=consent_key)
    if consent and not st.session_state.get(save_key, False):
        if st.button(
            "Lưu bản ghi",
            key=f"save_button_{source_hash}",
            icon=":material/save:",
            width="stretch",
        ):
            insert_prediction(
                image_name=image_name,
                source=source,
                predicted_label=result.label,
                predicted_name_vi=result.name_vi,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                prediction_id=str(uuid.uuid4()),
            )
            st.session_state[save_key] = True
            st.success("Đã lưu metadata kết quả. Ảnh không được lưu.", icon=":material/check_circle:")
    elif st.session_state.get(save_key, False):
        st.success("Bản ghi này đã được lưu vào lịch sử.", icon=":material/check_circle:")


def render() -> None:
    render_header(
        "Nhận diện rác thải",
        "Phân tích hình ảnh trực tiếp bằng mô hình MobileNetV2 đã huấn luyện",
        "📷",
    )

    st.markdown(
        "<div class='privacy-note'>🔒 Ảnh chỉ được xử lý trong phiên ứng dụng và không tự động lưu vào cơ sở dữ liệu.</div>",
        unsafe_allow_html=True,
    )

    upload_tab, camera_tab = st.tabs(["Tải ảnh", "Chụp ảnh"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Chọn ảnh JPG, JPEG hoặc PNG",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="waste_uploader",
            help="Tệp hợp lệ: JPG, JPEG, PNG.",
        )
    with camera_tab:
        captured = st.camera_input("Chụp một vật thể rác", key="waste_camera")

    source_file = captured if captured is not None else uploaded
    source = "camera" if captured is not None else "upload"
    if source_file is None:
        st.info(
            "Hãy tải ảnh hoặc chụp ảnh để bắt đầu nhận diện.",
            icon=":material/upload_file:",
        )
        return

    image_bytes = source_file.getvalue()
    source_hash = hashlib.sha256(image_bytes).hexdigest()[:20]
    image_name = getattr(source_file, "name", None) or f"camera_{source_hash}.jpg"

    try:
        preview = open_image(image_bytes)
    except ImageProcessingError as exc:
        st.error(str(exc), icon=":material/broken_image:")
        return

    render_section("Xem trước hình ảnh", "Ảnh đầu vào")
    preview_col, guide_col = st.columns([1.2, 1])
    with preview_col:
        st.image(
            preview,
            caption=f"Ảnh đã chọn: {image_name}",
            width="stretch",
            alt="Ảnh rác được chọn để nhận diện",
        )
    with guide_col:
        with st.container(border=True):
            st.markdown("#### Để có kết quả tốt hơn")
            st.markdown(
                "- Chụp rõ nét, đủ ánh sáng.\n"
                "- Đặt một vật thể chính ở giữa ảnh.\n"
                "- Hạn chế nền phức tạp hoặc nhiều loại rác cùng lúc.\n"
                "- Kết quả AI chỉ mang tính hỗ trợ."
            )

    result_key = f"prediction_{source_hash}"
    if st.button(
        "Nhận diện rác thải",
        type="primary",
        icon=":material/center_focus_strong:",
        width="stretch",
    ):
        try:
            with st.spinner("Mô hình đang phân tích ảnh...", show_time=True):
                metadata = load_metadata()
                model = load_model()
                st.session_state[result_key] = predict_waste(
                    image_bytes,
                    model=model,
                    metadata=metadata,
                )
        except (ModelConfigurationError, ModelLoadError, ImageProcessingError, PredictionError) as exc:
            st.error(f"Không thể nhận diện: {exc}", icon=":material/error:")
            return
        except Exception as exc:
            st.error(
                f"Đã xảy ra lỗi không mong đợi khi nhận diện: {exc}",
                icon=":material/error:",
            )
            return

    result = st.session_state.get(result_key)
    if result is not None:
        render_section("Kết quả phân tích", "MobileNetV2")
        _render_result(result, image_name, source, source_hash)
