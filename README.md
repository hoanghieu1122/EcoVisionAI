# EcoVision AI – Hệ thống nhận diện và phân loại rác thải thông minh

Website Streamlit phục vụ đồ án **“Xây dựng hệ thống nhận diện và phân loại rác thải sử dụng trí tuệ nhân tạo”**. Ứng dụng sử dụng **trực tiếp** mô hình `models/waste_model.keras` đã huấn luyện trên Google Colab bằng MobileNetV2 và TrashNet; không tải mô hình pretrained thay thế, không huấn luyện lại và không tạo kết quả giả.

## 1. Thông tin mô hình và nguồn dữ liệu

- Kiến trúc: MobileNetV2.
- Framework khi xuất: TensorFlow 2.21.0.
- Đầu vào: RGB, 224 × 224, resize bilinear.
- Tiền xử lý: `tensorflow.keras.applications.mobilenet_v2.preprocess_input` vì `model_has_builtin_preprocess=false`.
- Thứ tự nhãn cố định: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- Tập test độc lập: 380 ảnh.
- Test Accuracy: 84,21%.
- Macro F1-Score: 0,8270.

> 84,21% là kết quả trên tập test, không phải cam kết cho mọi ảnh thực tế. Xác suất softmax không phải độ chính xác thực tế.

## 2. Chức năng

1. **Trang chủ:** giới thiệu hệ thống, công nghệ, sáu loại rác và chỉ số thực nghiệm.
2. **Nhận diện AI:** tải JPG/JPEG/PNG hoặc chụp camera, xem ảnh, chủ động bấm nhận diện, nhận đủ sáu xác suất và hướng dẫn xử lý.
3. **Hướng dẫn phân loại:** mô tả, ví dụ, điều kiện tái chế và lưu ý thu gom cho sáu nhóm.
4. **Lịch sử nhận diện:** opt-in lưu metadata, tìm kiếm/lọc ngày, thống kê sử dụng, tải CSV và xóa có xác nhận.
5. **Thống kê & đánh giá:** đọc trực tiếp JSON/CSV/PNG trong `evaluation_results`, gồm confusion matrix, classification report và training curves.

Ứng dụng **không tự động lưu ảnh**. SQLite chỉ lưu ID, thời gian, tên file (basename), nguồn, nhãn, tên Việt, xác suất và thời gian xử lý khi người dùng chọn đồng ý.

## 3. Cấu trúc dự án

```text
EcoVision_AI/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── waste_model.keras
│   ├── class_names.json
│   └── model_config.json
├── views/
│   ├── home.py
│   ├── recognition.py
│   ├── waste_guide.py
│   ├── history.py
│   └── statistics.py
├── utils/
│   ├── paths.py
│   ├── ui.py
│   ├── model_loader.py
│   ├── prediction.py
│   ├── database.py
│   ├── evaluation.py
│   └── waste_information.py
├── assets/style.css
├── evaluation_results/
├── sample_test_images/
├── tests/
└── database/history.db       # tự tạo khi chạy
```

`views/` được dùng thay cho `pages/` để ứng dụng có sidebar tiếng Việt duy nhất, tránh Streamlit tự tạo thêm các trang trùng lặp.

## 4. Cài đặt trên Windows bằng VS Code

### Yêu cầu

- Windows 11 hoặc Windows 10 64-bit.
- Python **3.11 64-bit**.
- VS Code và extension Python.
- Gói mô hình đã giải nén đúng cấu trúc ở trên.

Mở thư mục dự án trong VS Code, mở **Terminal > New Terminal** (PowerShell), rồi chạy:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt, chạy một lần trong terminal hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Chọn interpreter `.venv` trong VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter**.

### Chạy website

```powershell
streamlit run app.py
```

Trình duyệt thường mở tại `http://localhost:8501`.

### Đổi ngưỡng cảnh báo

Mặc định ứng dụng cảnh báo khi xác suất cao nhất dưới 60%. Có thể đổi trước khi chạy:

```powershell
$env:ECOVISION_CONFIDENCE_THRESHOLD="0.65"
streamlit run app.py
```

Giá trị hợp lệ từ `0` đến `1`.

## 5. Chạy kiểm thử

```powershell
python -m pytest -v
```

Bộ test kiểm tra:

- đúng thứ tự sáu nhãn và metadata 224 × 224 RGB;
- preprocessing float32, MobileNetV2 và không xử lý hai lần;
- model thật tải được, có input/output đúng;
- dự đoán thật cả sáu ảnh mẫu (không ép kết quả phải đúng nhãn trước khi chạy);
- ảnh không hợp lệ và model không tồn tại;
- SQLite: lưu, chống trùng ID, lọc, CSV, xóa và không có bytes ảnh;
- dữ liệu đánh giá: 380 mẫu, metric, confusion matrix và 22 epoch.

Test camera tự động chỉ xác minh pipeline nhận bytes/file-like. Quyền camera và phần cứng cần kiểm tra thủ công trong trình duyệt.

## 6. Checklist trình diễn

- [ ] Chạy được `streamlit run app.py` và mở đủ 5 trang.
- [ ] Trang chủ ghi rõ 84,21% là độ chính xác trên tập test.
- [ ] Upload lần lượt 6 ảnh trong `sample_test_images/`; mỗi lần model trả đủ 6 xác suất.
- [ ] Không tuyên bố cả 6 ảnh phải được phân loại đúng nếu chưa quan sát kết quả.
- [ ] Chụp camera được sau khi cấp quyền cho trình duyệt.
- [ ] Ảnh hỏng hiển thị lỗi rõ ràng.
- [ ] Khi model bị thiếu/hỏng, hệ thống dừng và báo lỗi, không dự đoán giả.
- [ ] Cảnh báo xuất hiện khi confidence thấp hơn threshold.
- [ ] Không chọn đồng ý thì lịch sử không tăng.
- [ ] Chọn đồng ý và bấm lưu thì metadata xuất hiện trong lịch sử, không có file ảnh mới.
- [ ] Lọc nhãn/ngày, biểu đồ sử dụng, CSV và xóa lịch sử hoạt động.
- [ ] Trang đánh giá hiển thị đúng 84,21%, 0,8270, 380 mẫu, confusion matrix và training curves từ artifact thật.
- [ ] Giao diện kiểm tra trên cả cửa sổ desktop và kích thước mobile.

## 7. Xử lý lỗi thường gặp

- **Không tải được model:** kiểm tra `models/waste_model.keras`, Python 3.11 64-bit và TensorFlow 2.21.x. Ứng dụng không tự chuyển sang model khác.
- **TensorFlow không cài được:** xác nhận terminal đang dùng `.venv` và `python --version` là 3.11; nâng cấp pip rồi cài lại requirements.
- **Camera không hiện:** cấp quyền camera cho `localhost` trong trình duyệt, đóng ứng dụng khác đang chiếm camera và tải lại trang.
- **Dữ liệu đánh giá lỗi:** không sửa số liệu trong code; khôi phục nguyên thư mục `evaluation_results` từ ZIP.
- **CSV tiếng Việt:** file xuất dùng UTF-8 BOM để tương thích Excel Windows.

## 8. Nguồn gói mô hình

Các file model và đánh giá trong dự án được xuất trực tiếp từ Google Colab sau khi huấn luyện/đánh giá TrashNet. Gói gốc `trashnet_waste_classifier_package.zip` được giữ nguyên để đối chiếu provenance.
