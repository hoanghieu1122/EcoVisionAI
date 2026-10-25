"""Lưu metadata lịch sử nhận diện bằng SQLite; không lưu nội dung ảnh."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd

from utils.paths import DATABASE_PATH


HISTORY_COLUMNS = [
    "prediction_id",
    "created_at",
    "image_name",
    "source",
    "predicted_label",
    "predicted_name_vi",
    "confidence",
    "processing_time_ms",
]


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path = DATABASE_PATH) -> None:
    """Tạo bảng và index nếu chưa tồn tại."""
    with closing(_connect(db_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                image_name TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('upload', 'camera')),
                predicted_label TEXT NOT NULL,
                predicted_name_vi TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                processing_time_ms REAL NOT NULL CHECK (processing_time_ms >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_history_created_at
                ON prediction_history(created_at);
            CREATE INDEX IF NOT EXISTS idx_history_label
                ON prediction_history(predicted_label);
            """
        )
        connection.commit()


def _safe_image_name(image_name: str) -> str:
    cleaned = Path(image_name or "anh_khong_ten").name.strip()
    return cleaned[:255] or "anh_khong_ten"


def insert_prediction(
    image_name: str,
    source: str,
    predicted_label: str,
    predicted_name_vi: str,
    confidence: float,
    processing_time_ms: float,
    prediction_id: str | None = None,
    created_at: datetime | None = None,
    db_path: Path = DATABASE_PATH,
) -> str:
    """Lưu một bản ghi metadata; ID trùng sẽ không tạo bản sao."""
    if source not in {"upload", "camera"}:
        raise ValueError("source chỉ có thể là 'upload' hoặc 'camera'.")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence phải nằm trong khoảng 0 đến 1.")
    if float(processing_time_ms) < 0:
        raise ValueError("processing_time_ms không được âm.")

    initialize_database(db_path)
    record_id = prediction_id or str(uuid.uuid4())
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp_text = timestamp.astimezone(timezone.utc).isoformat(timespec="seconds")

    with closing(_connect(db_path)) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO prediction_history (
                prediction_id, created_at, image_name, source, predicted_label,
                predicted_name_vi, confidence, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                timestamp_text,
                _safe_image_name(image_name),
                source,
                predicted_label,
                predicted_name_vi,
                float(confidence),
                float(processing_time_ms),
            ),
        )
        connection.commit()
    return record_id


def _utc_boundary(value: date, end_of_day: bool = False) -> str:
    boundary_time = time.max if end_of_day else time.min
    return datetime.combine(value, boundary_time, tzinfo=timezone.utc).isoformat()


def get_history(
    label: str | None = None,
    search: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    db_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Truy vấn lịch sử bằng tham số, hỗ trợ nhãn/từ khóa/khoảng ngày UTC."""
    initialize_database(db_path)
    clauses: list[str] = []
    parameters: list[object] = []

    if label and label != "Tất cả":
        clauses.append("predicted_label = ?")
        parameters.append(label)
    if search:
        clauses.append("(image_name LIKE ? OR predicted_name_vi LIKE ? OR predicted_label LIKE ?)")
        term = f"%{search.strip()}%"
        parameters.extend([term, term, term])
    if start_date:
        clauses.append("created_at >= ?")
        parameters.append(_utc_boundary(start_date))
    if end_date:
        clauses.append("created_at <= ?")
        parameters.append(_utc_boundary(end_date, end_of_day=True))

    query = "SELECT " + ", ".join(HISTORY_COLUMNS) + " FROM prediction_history"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"

    with closing(_connect(db_path)) as connection:
        frame = pd.read_sql_query(query, connection, params=parameters)
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame


def history_statistics(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Đếm lượt sử dụng website theo nhãn AI."""
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        return pd.read_sql_query(
            """
            SELECT predicted_label, predicted_name_vi, COUNT(*) AS prediction_count,
                   AVG(confidence) AS average_confidence,
                   AVG(processing_time_ms) AS average_processing_time_ms
            FROM prediction_history
            GROUP BY predicted_label, predicted_name_vi
            ORDER BY prediction_count DESC, predicted_label
            """,
            connection,
        )


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Xuất CSV có BOM để Excel trên Windows đọc đúng tiếng Việt."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def clear_history(db_path: Path = DATABASE_PATH) -> int:
    """Xóa toàn bộ lịch sử và trả về số dòng đã xóa."""
    initialize_database(db_path)
    with closing(_connect(db_path)) as connection:
        cursor = connection.execute("DELETE FROM prediction_history")
        connection.commit()
        return int(cursor.rowcount)
