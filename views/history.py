"""Trang lịch sử nhận diện và thống kê sử dụng website."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.database import clear_history, dataframe_to_csv_bytes, get_history, history_statistics
from utils.ui import apply_plotly_theme, kpi_card, render_header, render_section
from utils.waste_information import CLASS_ORDER, WASTE_INFO


def _usage_figure(statistics: pd.DataFrame):
    """Tạo biểu đồ lượt sử dụng theo lớp với màu Emerald/Teal."""
    figure = px.bar(
        statistics,
        x="predicted_name_vi",
        y="prediction_count",
        labels={"predicted_name_vi": "Loại rác", "prediction_count": "Số lượt"},
        color="predicted_label",
        color_discrete_map={
            label: color
            for label, color in zip(
                CLASS_ORDER,
                ["#059669", "#10B981", "#0D9488", "#14B8A6", "#34D399", "#6EE7B7"],
            )
        },
        text="prediction_count",
    )
    figure.update_traces(
        textposition="outside",
        textfont={"color": "#0F172A"},
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="%{x}: %{y} lượt<extra></extra>",
    )
    figure.update_layout(bargap=0.34)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(rangemode="tozero", gridcolor="#E2E8F0")
    return apply_plotly_theme(
        figure,
        height=390,
        margin={"l": 42, "r": 28, "t": 28, "b": 72},
        show_legend=False,
    )


def render() -> None:
    render_header(
        "Lịch sử nhận diện",
        "Metadata kết quả chỉ được lưu khi người dùng chủ động đồng ý",
        "🕘",
    )
    st.caption("Cơ sở dữ liệu không lưu bytes ảnh hoặc bản sao ảnh cá nhân.")

    render_section("Tìm kiếm bản ghi", "Bộ lọc")
    filter_columns = st.columns([1.2, 1, 1, 1], vertical_alignment="bottom")
    with filter_columns[0]:
        search = st.text_input(
            "Tìm kiếm",
            placeholder="Tên ảnh, nhãn AI hoặc tên tiếng Việt",
            icon=":material/search:",
        )
    with filter_columns[1]:
        label_options = ["Tất cả"] + CLASS_ORDER
        label = st.selectbox(
            "Loại rác",
            label_options,
            format_func=lambda item: "Tất cả" if item == "Tất cả" else WASTE_INFO[item]["name_vi"],
        )
    default_start = date.today() - timedelta(days=30)
    with filter_columns[2]:
        start_date = st.date_input("Từ ngày", value=default_start)
    with filter_columns[3]:
        end_date = st.date_input("Đến ngày", value=date.today())

    if start_date > end_date:
        st.error("Ngày bắt đầu không được sau ngày kết thúc.", icon=":material/event_busy:")
        return

    try:
        frame = get_history(label=label, search=search, start_date=start_date, end_date=end_date)
        statistics = history_statistics()
        all_history = get_history()
    except Exception as exc:
        st.error(f"Không thể đọc lịch sử: {exc}", icon=":material/database:")
        return

    render_section("Tổng quan dữ liệu", "Chỉ số lịch sử")
    metric_columns = st.columns(3)
    with metric_columns[0]:
        kpi_card("Tổng lượt đã lưu", f"{len(all_history):,}", "#", "Toàn bộ lịch sử")
    with metric_columns[1]:
        kpi_card(
            "Xác suất trung bình",
            "—" if frame.empty else f"{frame['confidence'].mean():.1%}",
            "%",
            "Theo bộ lọc hiện tại",
        )
    with metric_columns[2]:
        kpi_card(
            "Thời gian trung bình",
            "—" if frame.empty else f"{frame['processing_time_ms'].mean():.1f} ms",
            "⏱",
            "Theo bộ lọc hiện tại",
        )

    render_section("Danh sách bản ghi", "Chi tiết")
    if frame.empty:
        st.info("Chưa có bản ghi phù hợp với bộ lọc.", icon=":material/inbox:")
    else:
        display = frame.copy()
        display["Thời gian nhận diện"] = (
            display["created_at"] + pd.Timedelta(hours=7)
        ).dt.strftime("%d/%m/%Y %H:%M:%S") + " ICT"
        display["Xác suất"] = display["confidence"]
        display["Thời gian xử lý"] = display["processing_time_ms"]
        display = display.rename(
            columns={
                "prediction_id": "Mã lượt",
                "image_name": "Tên ảnh",
                "predicted_name_vi": "Loại rác",
                "predicted_label": "Nhãn AI",
                "source": "Nguồn",
            }
        )[
            [
                "Mã lượt",
                "Thời gian nhận diện",
                "Tên ảnh",
                "Nguồn",
                "Loại rác",
                "Nhãn AI",
                "Xác suất",
                "Thời gian xử lý",
            ]
        ]
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "Mã lượt": st.column_config.TextColumn(width="small"),
                "Thời gian nhận diện": st.column_config.TextColumn(width="medium"),
                "Tên ảnh": st.column_config.TextColumn(width="medium"),
                "Nguồn": st.column_config.TextColumn(width="small"),
                "Loại rác": st.column_config.TextColumn(width="small"),
                "Nhãn AI": st.column_config.TextColumn(width="small"),
                "Xác suất": st.column_config.ProgressColumn(
                    format="percent",
                    min_value=0.0,
                    max_value=1.0,
                    width="small",
                ),
                "Thời gian xử lý": st.column_config.NumberColumn(
                    format="%.1f ms",
                    width="small",
                ),
            },
            alt="Danh sách metadata lịch sử nhận diện",
        )
        csv_display = display.copy()
        csv_display["Xác suất"] = csv_display["Xác suất"].map(lambda value: f"{value:.2%}")
        csv_display["Thời gian xử lý"] = csv_display["Thời gian xử lý"].map(
            lambda value: f"{value:.1f} ms"
        )
        st.download_button(
            "Xuất lịch sử CSV",
            data=dataframe_to_csv_bytes(csv_display),
            file_name="ecovision_lich_su.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

    render_section("Lượt sử dụng theo loại rác", "Thống kê website")
    st.caption("Biểu đồ này là số lượt người dùng đã đồng ý lưu, không phải dữ liệu đánh giá mô hình.")
    if statistics.empty:
        st.info("Chưa có dữ liệu sử dụng để thống kê.", icon=":material/bar_chart:")
    else:
        st.plotly_chart(
            _usage_figure(statistics),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            alt="Số lượt nhận diện đã lưu theo từng loại rác",
        )

    with st.expander("Xóa lịch sử", icon=":material/delete_forever:"):
        st.warning("Thao tác này xóa toàn bộ metadata lịch sử và không thể hoàn tác.")
        confirmed = st.checkbox(
            "Tôi hiểu và xác nhận muốn xóa toàn bộ lịch sử",
            key="confirm_clear_history",
        )
        if st.button(
            "Xóa toàn bộ lịch sử",
            type="secondary",
            icon=":material/delete:",
            disabled=not confirmed,
        ):
            deleted = clear_history()
            st.success(f"Đã xóa {deleted} bản ghi.", icon=":material/check_circle:")
            st.rerun()
