"""Nội dung hướng dẫn phân loại cho sáu nhãn của mô hình."""

CLASS_ORDER = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

WASTE_INFO = {
    "cardboard": {
        "name_vi": "Bìa carton",
        "icon": "📦",
        "color": "#A66A3F",
        "description": "Bao bì giấy nhiều lớp, thường dày và có cấu trúc sóng.",
        "examples": "Thùng giao hàng, hộp carton, tấm bìa đóng gói.",
        "recyclability": "Thường có thể tái chế khi sạch, khô và không dính nhiều dầu mỡ; quy định tiếp nhận khác nhau theo địa phương.",
        "handling": "Làm phẳng, tháo băng keo nếu thuận tiện và giữ khô. Carton dính thức ăn hoặc lớp phủ đặc biệt cần xử lý theo hướng dẫn địa phương.",
    },
    "glass": {
        "name_vi": "Thủy tinh",
        "icon": "🍾",
        "color": "#29A37A",
        "description": "Chai, lọ và vật dụng làm từ thủy tinh.",
        "examples": "Chai đồ uống, lọ thực phẩm, hũ thủy tinh.",
        "recyclability": "Nhiều chai lọ có thể tái chế, nhưng kính cửa, gương, gốm và thủy tinh chịu nhiệt thường không cùng luồng thu gom.",
        "handling": "Đổ sạch chất còn lại, súc nhẹ nếu phù hợp và tháo nắp. Bọc an toàn đồ vỡ, ghi chú rõ và tuân thủ điểm thu gom địa phương.",
    },
    "metal": {
        "name_vi": "Kim loại",
        "icon": "🥫",
        "color": "#607D8B",
        "description": "Vật dụng kim loại phổ biến như nhôm hoặc thép.",
        "examples": "Lon nước, hộp thiếc, nắp kim loại, giấy nhôm sạch.",
        "recyclability": "Thường có giá trị tái chế cao nếu được cơ sở địa phương tiếp nhận và không lẫn chất nguy hại.",
        "handling": "Làm sạch phần thực phẩm, để ráo và ép gọn nếu nơi thu gom cho phép. Không đưa pin, bình xịt còn áp suất hay vật sắc nhọn vào luồng thông thường.",
    },
    "paper": {
        "name_vi": "Giấy",
        "icon": "📄",
        "color": "#4C8BF5",
        "description": "Sản phẩm giấy mỏng, khác với bìa carton dày.",
        "examples": "Báo, giấy in, vở, túi giấy, tờ rơi.",
        "recyclability": "Có thể tái chế khi sạch và khô. Giấy ướt, giấy vệ sinh, giấy nhiệt hoặc giấy dính nhiều dầu mỡ có thể không được chấp nhận.",
        "handling": "Giữ khô, loại bỏ vật liệu không phải giấy và phân loại theo hướng dẫn của đơn vị thu gom tại nơi bạn sinh sống.",
    },
    "plastic": {
        "name_vi": "Nhựa",
        "icon": "🧴",
        "color": "#EF8C2F",
        "description": "Bao bì và vật dụng bằng nhiều loại nhựa khác nhau.",
        "examples": "Chai nhựa, hộp đựng, can, cốc và bao bì nhựa.",
        "recyclability": "Phụ thuộc ký hiệu nhựa, hình dạng và năng lực tái chế địa phương; không phải mọi loại nhựa đều được tiếp nhận.",
        "handling": "Kiểm tra ký hiệu tái chế, làm sạch bao bì nếu phù hợp, để ráo và phân loại theo quy định thu gom tại địa phương.",
    },
    "trash": {
        "name_vi": "Rác khác",
        "icon": "🗑️",
        "color": "#6B7280",
        "description": "Vật không thuộc năm nhóm vật liệu còn lại hoặc khó tái chế trong luồng phổ thông.",
        "examples": "Bao bì hỗn hợp, vật bẩn, đồ dùng khó xác định vật liệu.",
        "recyclability": "Khả năng tái chế tùy thành phần. Chất thải điện tử, pin, hóa chất và y tế phải đi theo luồng chuyên biệt.",
        "handling": "Kiểm tra thành phần và hướng dẫn địa phương trước khi bỏ. Không trộn chất thải nguy hại vào rác sinh hoạt; mang tới điểm tiếp nhận chuyên dụng.",
    },
}


def get_waste_info(label: str) -> dict:
    """Trả về thông tin hướng dẫn của nhãn hợp lệ."""
    if label not in WASTE_INFO:
        raise KeyError(f"Nhãn rác không được hỗ trợ: {label}")
    return WASTE_INFO[label]


def vietnamese_name(label: str) -> str:
    """Ánh xạ nhãn AI sang tên tiếng Việt."""
    return get_waste_info(label)["name_vi"]
