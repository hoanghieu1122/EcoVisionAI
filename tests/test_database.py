from datetime import date, datetime, timezone

from utils.database import (
    clear_history,
    dataframe_to_csv_bytes,
    get_history,
    history_statistics,
    insert_prediction,
)


def test_database_full_workflow(tmp_path):
    db_path = tmp_path / "history.db"
    record_id = "fixed-id"
    created = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)

    insert_prediction(
        image_name="C:/private/folder/chai.png",
        source="upload",
        predicted_label="plastic",
        predicted_name_vi="Nhựa",
        confidence=0.925,
        processing_time_ms=42.5,
        prediction_id=record_id,
        created_at=created,
        db_path=db_path,
    )
    insert_prediction(
        image_name="chai.png",
        source="upload",
        predicted_label="plastic",
        predicted_name_vi="Nhựa",
        confidence=0.925,
        processing_time_ms=42.5,
        prediction_id=record_id,
        created_at=created,
        db_path=db_path,
    )

    frame = get_history(
        label="plastic",
        search="chai",
        start_date=date(2026, 10, 24),
        end_date=date(2026, 10, 24),
        db_path=db_path,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["image_name"] == "chai.png"
    assert "image_bytes" not in frame.columns

    stats = history_statistics(db_path)
    assert stats.iloc[0]["prediction_count"] == 1

    csv_data = dataframe_to_csv_bytes(frame)
    assert csv_data.startswith(b"\xef\xbb\xbf")
    assert b"private/folder" not in csv_data

    assert clear_history(db_path) == 1
    assert get_history(db_path=db_path).empty
