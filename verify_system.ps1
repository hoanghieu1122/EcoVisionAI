$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python311 = "$env:LOCALAPPDATA\Python\pythoncore-3.11-64\python.exe"
if (-not (Test-Path $Python311)) {
    throw "Không tìm thấy Python 3.11 tại $Python311"
}

Write-Host "=== Python ==="
& $Python311 --version

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "=== Tạo virtual environment ==="
    & $Python311 -m venv .venv
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Write-Host "=== Cài dependencies ==="
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host "=== Chạy pytest ==="
& $VenvPython -m pytest -v

Write-Host "=== Dự đoán 6 ảnh mẫu bằng model thật ==="
& $VenvPython verify_predictions.py

Write-Host "=== Hoàn tất kiểm tra tự động ==="
Write-Host "Chạy website bằng: .\.venv\Scripts\streamlit.exe run app.py"
