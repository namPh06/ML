"""KNN truy xuất tương đồng: scaler toàn kho train và chỉ mục riêng cho từng trạm."""
from numbers import Integral
from pathlib import Path
from collections.abc import Mapping
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .data import FEATURES, numeric_features

class SimilaritySearch:
    def __init__(self, features=None, metric="euclidean"):
        self.features = list(FEATURES if features is None else features)
        if not self.features or len(set(self.features)) != len(self.features) or set(self.features) & {"datetime", "station", "rank", "distance"}:
            raise ValueError("Danh sách đặc trưng không hợp lệ.")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric phải là euclidean hoặc manhattan.")
        self.metric = metric

    def fit(self, reference):
        required = ["datetime", "station", *self.features]
        if not all(c in reference for c in required) or reference.empty:
            raise ValueError(f"Kho lịch sử không được rỗng và cần các cột {required}.")
        rows = reference[required].copy().reset_index(drop=True)
        rows["datetime"] = pd.to_datetime(rows["datetime"], errors="raise")
        rows["station"] = rows["station"].astype("string").str.strip()
        if rows.datetime.dt.tz is not None:
            raise ValueError("Thời gian kho lịch sử cần là giờ địa phương không có timezone.")
        if rows[["datetime", "station"]].isna().any().any() or rows.station.eq("").any() or rows.duplicated(["station", "datetime"]).any():
            raise ValueError("Định danh bản ghi thiếu hoặc trùng.")
        values = numeric_features(rows, self.features)
        if values.isna().any().any():
            raise ValueError("Kho lịch sử có đặc trưng thiếu, vô hạn hoặc sai miền giá trị.")
        rows[self.features] = values
        self.rows = rows
        # Một scaler chung cho các trạm; tuyệt đối không fit lại theo truy vấn.
        self.scaler = StandardScaler().fit(values)
        self.matrix = self.scaler.transform(values)
        self.index = NearestNeighbors(metric=self.metric, algorithm="auto").fit(self.matrix)
        self.station_indices = {}
        for name, indices in rows.groupby("station", sort=True).indices.items():
            positions = np.asarray(indices)
            index = NearestNeighbors(metric=self.metric, algorithm="auto").fit(self.matrix[positions])
            self.station_indices[str(name)] = (positions, index)
        self.reference_max_time = rows.datetime.max()
        self.artifact_version = 2
        return self

    def find_similar_periods(self, input_data, k=5, station=None, past_only=True):
        """station=None: theo query; station='ALL': toàn kho; tên trạm: lọc trạm.

        Với datetime và past_only=True, truy vấn phải sau kho train để scaler
        không học từ tương lai. past_only=False chỉ dành cho duyệt kho.
        """
        if not hasattr(self, "index"):
            raise ValueError("Cần fit hoặc load bộ tìm kiếm trước.")
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k phải là số nguyên dương.")
        if isinstance(input_data, pd.Series):
            input_data = input_data.to_dict()
        if not isinstance(input_data, Mapping):
            raise ValueError("Đầu vào cần là dictionary hoặc pandas Series của một bản ghi.")
        query = dict(input_data)
        missing = [f for f in self.features if f not in query]
        if missing:
            raise ValueError(f"Đầu vào thiếu đặc trưng: {missing}")
        x = numeric_features(pd.DataFrame([{f: query[f] for f in self.features}]), self.features)
        if x.isna().any().any():
            raise ValueError("Đặc trưng phải là số hữu hạn; nồng độ/gió không âm, áp suất dương.")
        time = None
        if query.get("datetime") is not None:
            try:
                time = pd.Timestamp(query["datetime"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Thời gian truy vấn không hợp lệ.") from exc
            if pd.isna(time) or time.tzinfo is not None:
                raise ValueError("Thời gian truy vấn cần hợp lệ và không có timezone.")
            if past_only and time <= self.reference_max_time:
                raise ValueError("Truy vấn quá khứ phải sau toàn bộ kho train. Fit kho sớm hơn, hoặc dùng past_only=False để duyệt kho.")
        target = station if station is not None else query.get("station")
        if target is not None and not isinstance(target, str):
            raise ValueError("station phải là tên trạm, ALL hoặc None.")
        if target is None or target == "ALL":
            positions, index = np.arange(len(self.rows)), self.index
        else:
            if not isinstance(target, str) or target not in self.station_indices:
                raise ValueError(f"Trạm không có trong kho lịch sử: {target}")
            positions, index = self.station_indices[target]
        # Trong chế độ quá khứ, toàn kho đã nằm trước truy vấn. Không xây lại chỉ mục.
        if time is not None and not past_only:
            if query.get("station") is None:
                raise ValueError("Duyệt theo datetime cần station để loại đúng bản ghi truy vấn.")
            candidate = self.rows.iloc[positions]
            identity = candidate.datetime.eq(time) & candidate.station.eq(query["station"])
            if identity.any():
                positions = positions[~identity.to_numpy()]
                index = None
        if len(positions) < k:
            raise ValueError(f"Chỉ có {len(positions)} bản ghi phù hợp, không đủ k={k}.")
        q = self.scaler.transform(x[self.features])
        if index is None:
            # Ít dùng: duyệt một bản ghi có sẵn. Tính trực tiếp để khỏi fit lại cây.
            delta = np.abs(self.matrix[positions]-q[0])
            all_distances = np.sqrt((delta**2).sum(axis=1)) if self.metric == "euclidean" else delta.sum(axis=1)
            nearest = np.argsort(all_distances, kind="stable")[:k]
            selected, distance = positions[nearest], all_distances[nearest]
        else:
            distances, indices = index.kneighbors(q, n_neighbors=int(k))
            selected, distance = positions[indices[0]], distances[0]
        result = self.rows.iloc[selected].copy().reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, k+1))
        result["distance"] = distance
        return result

    def save(self, destination):
        if not hasattr(self, "index"):
            raise ValueError("Cần fit trước khi lưu.")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            joblib.dump(self, temporary, compress=3)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, source):
        model = joblib.load(source)
        if not isinstance(model, cls) or getattr(model, "artifact_version", None) != 2:
            raise ValueError("Artifact cũ hoặc không đúng định dạng. Chạy lại python -m module3.train.")
        return model

def find_similar_periods(input_data, k=5, *, searcher=None, model_path=None, **kwargs):
    """API ghép nhóm: nạp searcher một lần hoặc truyền model_path."""
    if searcher is None:
        if model_path is None:
            raise ValueError("Cần truyền searcher hoặc model_path.")
        searcher = SimilaritySearch.load(model_path)
    return searcher.find_similar_periods(input_data, k=k, **kwargs)
