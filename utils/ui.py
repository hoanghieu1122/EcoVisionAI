"""Thành phần giao diện dùng chung."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from utils.paths import STYLE_PATH

PLOTLY_FONT = "Plus Jakarta Sans, Inter, sans-serif"


def load_css() -> None:
    """Nạp CSS tùy chỉnh nếu tệp tồn tại."""
    if STYLE_PATH.is_file():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🌿") -> None:
    """Hiển thị tiêu đề trang nhất quán."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-icon" aria-hidden="true">{html.escape(icon)}</div>
            <div>
                <h1>{html.escape(title)}</h1>
                <p>{html.escape(subtitle)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, eyebrow: str | None = None) -> None:
    """Hiển thị nhãn và tiêu đề của một khu vực nội dung."""
    eyebrow_html = (
        f'<div class="section-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    )
    st.markdown(
        f'{eyebrow_html}<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def info_card(icon: str, title: str, body: str, css_class: str = "") -> None:
    """Hiển thị card thông tin tĩnh đã escape nội dung."""
    st.markdown(
        f"""
        <div class="info-card {html.escape(css_class)}">
            <div class="card-icon" aria-hidden="true">{html.escape(icon)}</div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def guide_card(icon: str, title: str, label: str, description: str) -> None:
    """Hiển thị card hướng dẫn phân loại tĩnh."""
    st.markdown(
        f"""
        <div class="guide-card">
            <div class="card-icon" aria-hidden="true">{html.escape(icon)}</div>
            <div class="guide-label">Nhãn AI · {html.escape(label)}</div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, icon: str, note: str = "") -> None:
    """Hiển thị KPI tĩnh với giá trị nguyên bản do bên gọi cung cấp."""
    note_html = f'<div class="kpi-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-label">{html.escape(label)}</div>
                <div class="kpi-icon" aria-hidden="true">{html.escape(icon)}</div>
            </div>
            <div class="kpi-value">{html.escape(value)}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Hiển thị metadata mô hình thật ở cuối sidebar."""
    st.markdown(
        """
        <div class="model-card">
          <div class="model-card-head"><span class="model-status"></span> Mô hình đang sẵn sàng</div>
          <div class="model-meta"><span>Kiến trúc</span><strong>MobileNetV2</strong></div>
          <div class="model-meta"><span>Số lớp</span><strong>6 lớp</strong></div>
          <div class="model-meta"><span>Đầu vào</span><strong>224 × 224 RGB</strong></div>
          <div class="model-meta"><span>Test Accuracy</span><strong>84,21%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(
    figure,
    *,
    height: int | None = None,
    margin: dict[str, int] | None = None,
    show_legend: bool | None = None,
):
    """Áp dụng giao diện Plotly sáng, tương phản cao và nhất quán."""
    layout: dict = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": PLOTLY_FONT, "color": "#0F172A", "size": 13},
        "hoverlabel": {
            "bgcolor": "#FFFFFF",
            "bordercolor": "#E2E8F0",
            "font": {"family": PLOTLY_FONT, "color": "#0F172A"},
        },
        "margin": margin or {"l": 24, "r": 24, "t": 42, "b": 36},
    }
    if height is not None:
        layout["height"] = height
    if show_legend is not None:
        layout["showlegend"] = show_legend
    figure.update_layout(**layout)
    figure.update_xaxes(
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
        gridcolor="#E2E8F0",
        zeroline=False,
        linecolor="#CBD5E1",
        automargin=True,
    )
    figure.update_yaxes(
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
        gridcolor="#E2E8F0",
        zeroline=False,
        linecolor="#CBD5E1",
        automargin=True,
    )
    return figure


def emerald_scale() -> Sequence[tuple[float, str]]:
    """Thang màu Emerald dùng cho dữ liệu liên tục."""
    return (
        (0.0, "#ECFDF5"),
        (0.2, "#D1FAE5"),
        (0.4, "#A7F3D0"),
        (0.6, "#34D399"),
        (0.8, "#059669"),
        (1.0, "#064E3B"),
    )


def navigate_to(page_name: str) -> None:
    """Chuyển trang bằng session state hiện có."""
    st.session_state["current_page"] = page_name
    st.rerun()
