"""Trang chủ EcoVision AI."""

import streamlit as st

from utils.ui import info_card, kpi_card, navigate_to, render_section
from utils.waste_information import CLASS_ORDER, WASTE_INFO


def render() -> None:
    st.markdown(
        """
        <section class="hero">
          <span class="hero-badge">🌱 Công nghệ AI vì môi trường</span>
          <h1>Nhận diện rác thải.<br><span>Phân loại thông minh.</span></h1>
          <p>EcoVision AI sử dụng mô hình MobileNetV2 đã huấn luyện trên TrashNet để phân tích hình ảnh và hỗ trợ bạn xử lý rác đúng cách hơn.</p>
          <div class="hero-stats">
            <div class="hero-stat"><strong>84,21%</strong><span>Accuracy tập test</span></div>
            <div class="hero-stat"><strong>6 nhóm</strong><span>Vật liệu hỗ trợ</span></div>
            <div class="hero-stat"><strong>224 × 224</strong><span>Đầu vào RGB</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    action_col, note_col = st.columns([1, 2], vertical_alignment="center")
    with action_col:
        if st.button(
            "Bắt đầu nhận diện",
            type="primary",
            icon=":material/photo_camera:",
            width="stretch",
        ):
            navigate_to("Nhận diện AI")
    with note_col:
        st.caption(
            "Tải ảnh JPG, JPEG, PNG hoặc chụp bằng camera. Ảnh không tự động được lưu."
        )

    render_section("Quy trình nhận diện trong ba bước", "Bắt đầu nhanh")
    columns = st.columns(3)
    with columns[0]:
        info_card("📸", "1. Chọn hình ảnh", "Tải ảnh từ thiết bị hoặc chụp trực tiếp bằng camera.")
    with columns[1]:
        info_card(
            "🧠",
            "2. AI phân tích",
            "Ảnh được tiền xử lý đúng cấu hình rồi đưa vào mô hình MobileNetV2.",
        )
    with columns[2]:
        info_card(
            "♻️",
            "3. Xử lý phù hợp",
            "Xem đủ sáu xác suất và gợi ý phân loại theo từng nhóm vật liệu.",
        )

    render_section("Sáu nhóm vật liệu được hỗ trợ", "Phạm vi nhận diện")
    rows = [st.columns(3), st.columns(3)]
    for index, label in enumerate(CLASS_ORDER):
        info = WASTE_INFO[label]
        with rows[index // 3][index % 3]:
            st.markdown(
                f'<div class="waste-pill"><span aria-hidden="true">{info["icon"]}</span><span>{info["name_vi"]}</span></div>',
                unsafe_allow_html=True,
            )

    render_section("Hiệu quả thực nghiệm của mô hình", "Kết quả đánh giá thật")
    metric_columns = st.columns(3)
    with metric_columns[0]:
        kpi_card("Độ chính xác tập test", "84,21%", "✓", "Đọc từ artifact đánh giá")
    with metric_columns[1]:
        kpi_card("Macro F1-Score", "0,8270", "F1", "Trung bình trên 6 lớp")
    with metric_columns[2]:
        kpi_card("Số mẫu kiểm thử", "380", "#", "Tập test độc lập")
    st.markdown(
        """
        <div class="disclaimer">
        <strong>Lưu ý:</strong> 84,21% là độ chính xác đo trên tập kiểm thử độc lập của mô hình,
        không phải cam kết độ chính xác cho mọi ảnh thực tế. Xác suất mô hình trả về cũng không đồng nghĩa
        với độ chính xác thực tế.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section("MobileNetV2 tối ưu cho trải nghiệm nhanh", "Công nghệ lõi")
    st.write(
        "MobileNetV2 là kiến trúc mạng nơ-ron tích chập tối ưu cho thiết bị có tài nguyên giới hạn. "
        "EcoVision AI sử dụng trực tiếp mô hình `.keras` đã được huấn luyện và đóng gói; ứng dụng không "
        "tải MobileNetV2 mặc định, không huấn luyện lại và không tạo kết quả giả."
    )
