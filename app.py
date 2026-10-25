"""Điểm khởi chạy website EcoVision AI."""

from __future__ import annotations

import streamlit as st

from utils.database import initialize_database
from utils.ui import load_css, render_sidebar_footer
from views import history, home, recognition, statistics, waste_guide

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

PAGES = {
    "Trang chủ": ("⌂", home.render),
    "Nhận diện AI": ("◉", recognition.render),
    "Hướng dẫn phân loại": ("♻", waste_guide.render),
    "Lịch sử nhận diện": ("◷", history.render),
    "Thống kê & đánh giá": ("▥", statistics.render),
}

if "current_page" not in st.session_state or st.session_state["current_page"] not in PAGES:
    st.session_state["current_page"] = "Trang chủ"

try:
    initialize_database()
except Exception as exc:
    st.error(f"Không thể khởi tạo cơ sở dữ liệu lịch sử: {exc}", icon=":material/error:")

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark" aria-hidden="true">🌿</div>
          <div>
            <div class="brand-name">EcoVision AI</div>
            <div class="brand-sub">Phân loại rác thông minh</div>
          </div>
        </div>
        <div class="sidebar-section-label">Không gian làm việc</div>
        """,
        unsafe_allow_html=True,
    )
    page_names = list(PAGES)
    current_index = page_names.index(st.session_state["current_page"])
    selected = st.radio(
        "Điều hướng",
        options=page_names,
        index=current_index,
        format_func=lambda name: f"{PAGES[name][0]}  {name}",
        label_visibility="collapsed",
        width="stretch",
    )
    st.session_state["current_page"] = selected
    render_sidebar_footer()

try:
    PAGES[st.session_state["current_page"]][1]()
except Exception as exc:
    st.error(
        "Trang không thể hiển thị do lỗi không mong đợi.",
        icon=":material/error:",
    )
    with st.expander("Chi tiết kỹ thuật", icon=":material/code:"):
        st.exception(exc)
