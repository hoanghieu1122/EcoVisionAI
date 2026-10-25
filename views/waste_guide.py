"""Trang hướng dẫn phân loại sáu nhóm rác."""

import streamlit as st

from utils.ui import guide_card, render_header, render_section
from utils.waste_information import CLASS_ORDER, WASTE_INFO


def render() -> None:
    render_header(
        "Hướng dẫn phân loại rác",
        "Thông tin tham khảo cho sáu nhóm vật liệu mà mô hình nhận diện",
        "♻️",
    )
    st.warning(
        "Quy định thu gom và năng lực tái chế khác nhau theo từng địa phương. Hãy ưu tiên hướng dẫn của đơn vị thu gom nơi bạn sinh sống.",
        icon=":material/location_on:",
    )

    render_section("Tra cứu theo từng nhóm vật liệu", "Cẩm nang phân loại")
    for row_start in range(0, len(CLASS_ORDER), 2):
        columns = st.columns(2)
        for column, label in zip(columns, CLASS_ORDER[row_start : row_start + 2]):
            info = WASTE_INFO[label]
            with column:
                guide_card(info["icon"], info["name_vi"], label, info["description"])
                with st.expander(
                    f"Chi tiết về {info['name_vi']}",
                    icon=":material/info:",
                ):
                    st.write(f"**Ví dụ phổ biến:** {info['examples']}")
                    st.write(f"**Khả năng tái chế:** {info['recyclability']}")
                    st.write(f"**Cách xử lý:** {info['handling']}")
        st.write("")

    st.info(
        "Pin, thiết bị điện tử, hóa chất, bình chịu áp suất, bóng đèn và chất thải y tế cần điểm tiếp nhận chuyên dụng; không dựa riêng vào kết quả nhận diện hình ảnh để quyết định cách xử lý.",
        icon=":material/health_and_safety:",
    )
