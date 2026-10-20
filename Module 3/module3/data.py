"""Dữ liệu dùng chung cho module 3; thời gian là giờ địa phương Bắc Kinh."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

FEATURES = ["PM2.5", "PM10", "TEMP", "PRES", "DEWP", "WSPM"]
UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "TEMP": "°C", "PRES": "hPa", "DEWP": "°C", "WSPM": "m/s"}
TRAIN_END = "2015-11-01"
VALID_END = "2016-07-01"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality"

def read_data(source):
    """Đọc CSV, thư mục PRSA hoặc ZIP UCI (kể cả ZIP lồng), không giải nén ra đĩa."""
    source = Path(source)
    def read_zip(blob):
        parts = []
        with ZipFile(blob) as archive:
            for name in sorted(archive.namelist()):
                if Path(name).name.startswith("PRSA_Data_") and name.lower().endswith(".csv"):
                    parts.append(pd.read_csv(BytesIO(archive.read(name))))
                elif name.lower().endswith(".zip"):
                    parts.extend(read_zip(BytesIO(archive.read(name))))
        return parts
    if source.is_dir():
        parts = [pd.read_csv(p) for p in sorted(source.rglob("PRSA_Data_*.csv"))]
    elif source.suffix.lower() == ".zip":
        parts = read_zip(source)
    else:
        parts = [pd.read_csv(source)]
    if not parts:
        raise ValueError("Không tìm thấy CSV PRSA trong nguồn dữ liệu.")
    return pd.concat(parts, ignore_index=True)

def numeric_features(frame, features):
    values = frame[features].apply(pd.to_numeric, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    for col in features:
        if col in {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "WSPM", "RAIN"}:
            values.loc[values[col] < 0, col] = np.nan
        elif col == "PRES":
            values.loc[values[col] <= 0, col] = np.nan
    return values

def prepare_data(raw, station=None, features=None):
    """Mặc định giữ tất cả trạm; bỏ hàng thiếu/không hợp lệ, không điền từ tương lai."""
    features = list(FEATURES if features is None else features)
    if not features or len(features) != len(set(features)) or set(features) & {"datetime", "station", "rank", "distance"}:
        raise ValueError("Danh sách đặc trưng không hợp lệ.")
    df = raw.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in ["station", *features] if c not in df]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    input_rows = len(df)
    if "datetime" not in df:
        time_cols = ["year", "month", "day", "hour"]
        if not all(c in df for c in time_cols):
            raise ValueError("Cần datetime hoặc year/month/day/hour.")
        df["datetime"] = pd.to_datetime(df[time_cols], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        raise ValueError("Thời gian phải thống nhất, không trộn múi giờ.")
    if df["datetime"].dt.tz is not None:
        raise ValueError("Hãy chuyển datetime về giờ địa phương và bỏ timezone trước khi tích hợp.")
    df["station"] = df["station"].astype("string").str.strip()
    valid_keys = df["datetime"].notna() & df["station"].notna() & df["station"].ne("")
    invalid_keys = int((~valid_keys).sum())
    df = df[valid_keys].copy()
    if station is not None and station != "ALL":
        df = df[df["station"].eq(station)].copy()
    if df.empty:
        raise ValueError("Không có dữ liệu cho trạm đã chọn.")
    selected_rows = len(df)
    values = numeric_features(df, features)
    invalid_counts = {col: int((df[col].notna() & values[col].isna()).sum()) for col in features}
    df[features] = values
    keys = ["station", "datetime"]
    unique = df.drop_duplicates(subset=keys + features)
    if unique.duplicated(keys).any():
        raise ValueError("Trùng station/datetime nhưng khác đặc trưng. Nhóm cần xử lý xung đột trước.")
    duplicates = len(df)-len(unique)
    df = unique.sort_values(["datetime", "station"]).reset_index(drop=True)
    quality = []
    for name, group in df.groupby("station", sort=True):
        incomplete = group[features].isna().any(axis=1)
        quality.append({"station": str(name), "rows_before_missing_filter": len(group), "rows_used": int((~incomplete).sum()), "rows_dropped_missing": int(incomplete.sum()), **{f"missing_{c}": int(group[c].isna().sum()) for c in features}})
    report = {"input_rows": input_rows, "invalid_identity_rows": invalid_keys, "selected_rows": selected_rows, "duplicates_removed": duplicates, "rows_before_missing_filter": len(df), "missing_by_feature": {c: int(df[c].isna().sum()) for c in features}, "invalid_by_feature": invalid_counts, "by_station": quality}
    df = df.dropna(subset=features).reset_index(drop=True)
    if df.empty:
        raise ValueError("Không còn bản ghi đủ đặc trưng sau xử lý thiếu.")
    report.update(rows_used=len(df), rows_dropped_missing=report["rows_before_missing_filter"]-len(df), stations=sorted(df.station.unique().tolist()))
    return df, report

def split_by_time(df, train_end=TRAIN_END, valid_end=VALID_END):
    a, b = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    if pd.isna(a) or pd.isna(b) or a.tzinfo is not None or b.tzinfo is not None or a >= b:
        raise ValueError("Mốc thời gian phải hợp lệ, không có timezone; train_end phải trước valid_end.")
    t = df["datetime"]
    return (df[t < a].copy(), df[(t >= a) & (t < b)].copy(), df[t >= b].copy())
