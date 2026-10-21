"""Ứng dụng dự đoán; mô hình được huấn luyện trong module1.ipynb."""
from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "linear_regression.joblib"


def predict_pm25(input_data, model_path=MODEL_PATH):
    """Nhận một bản ghi tại giờ t, trả PM2.5 dự đoán tại t+1 (µg/m³)."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Chưa có mô hình. Hãy chạy python run_module1.py trước.")
    model_info = joblib.load(model_path)
    pipeline = model_info["pipeline"]
    numeric_features = model_info["numeric_features"]
    if isinstance(input_data, pd.Series):
        input_data = input_data.to_dict()
    if not isinstance(input_data, Mapping):
        raise ValueError("Đầu vào phải là dictionary hoặc pandas Series.")
    required = ["station", "datetime", *numeric_features]
    missing = [name for name in required if name not in input_data]
    if missing:
        raise ValueError(f"Thiếu đầu vào: {missing}")
    station = input_data["station"]
    if not isinstance(station, str) or not station.strip():
        raise ValueError("station phải là tên trạm không rỗng.")
    try:
        time = pd.Timestamp(input_data["datetime"])
    except (ValueError, TypeError) as exc:
        raise ValueError("datetime không hợp lệ.") from exc
    if pd.isna(time) or time.tzinfo is not None or time != time.floor("h"):
        raise ValueError("datetime phải là giờ tròn tại Bắc Kinh, không gắn timezone.")
    record = {"station": station.strip()}
    for name in numeric_features:
        value = input_data[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} phải là số, không phải đúng/sai.")
        try:
            value = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} phải là số.") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} phải hữu hạn, không được thiếu.")
        if name not in {"TEMP", "DEWP", "PRES"} and value < 0:
            raise ValueError(f"{name} không được âm.")
        if name == "PRES" and value <= 0:
            raise ValueError("PRES phải lớn hơn 0 hPa.")
        record[name] = value
    record.update(hour=time.hour, day_of_week=time.dayofweek, month=time.month)
    frame = pd.DataFrame([record])[model_info["features"]]
    encoder = pipeline.named_steps["preprocess"].named_transformers_["category"]
    for name, categories in zip(encoder.feature_names_in_, encoder.categories_):
        if frame.iloc[0][name] not in categories:
            raise ValueError(f"{name}={frame.iloc[0][name]} chưa xuất hiện trong tập huấn luyện.")
    return float(pipeline.predict(frame)[0])


LABELS = {
    "PM2.5": "PM2.5 hiện tại (µg/m³)", "PM10": "PM10 hiện tại (µg/m³)",
    "TEMP": "Nhiệt độ (°C)", "PRES": "Áp suất (hPa)",
    "DEWP": "Điểm sương (°C)", "WSPM": "Tốc độ gió (m/s)",
    "PM25_lag_1": "PM2.5 cách 1 giờ (µg/m³)",
    "PM25_lag_2": "PM2.5 cách 2 giờ (µg/m³)",
    "PM25_lag_3": "PM2.5 cách 3 giờ (µg/m³)",
}


@st.cache_data
def load_results(output_path, version):
    """Đọc lại khi mô hình đổi sau một lần chạy notebook."""
    output = Path(output_path)
    metadata = joblib.load(output / "linear_regression.joblib")
    metrics = pd.read_csv(output / "metrics.csv")
    predictions = pd.read_csv(output / "predictions.csv", parse_dates=["datetime", "target_datetime"])
    samples = predictions[predictions.split.eq("test")].groupby("station", sort=True).head(168)
    coefficients = pd.read_csv(output / "coefficients.csv")
    return metadata, metrics, predictions, samples, coefficients


def render_prediction(output, metadata, samples, version):
    NUMERIC_FEATURES = metadata["numeric_features"]
    UNITS = metadata["units"]
    st.subheader("Dự đoán PM2.5 của giờ tiếp theo")
    st.caption("Đầu vào tại giờ t và ba giờ trước, cùng một trạm. Thời gian dùng giờ Bắc Kinh UTC+8.")
    mode = st.radio("Cách nhập", ["Chọn mẫu test", "Nhập điều kiện"],
                    horizontal=True, key="prediction_mode")
    station = st.selectbox("Trạm quan trắc", metadata["stations"], key="prediction_station")
    station_samples = samples[samples.station.eq(station)]
    default = station_samples.iloc[0]
    if mode == "Chọn mẫu test":
        st.caption("Chọn trong 168 mẫu đầu tiên của mỗi trạm thuộc tập test; tất cả mẫu đều có giá trị thực tế để đối chiếu.")
        selected = st.selectbox("Giờ quan trắc t", station_samples.datetime.tolist(),
                                format_func=lambda value: value.strftime("%d/%m/%Y %H:%M"), key="sample_time")
        record = station_samples[station_samples.datetime.eq(selected)].iloc[0].to_dict()
        with st.expander("Xem đầu vào của mẫu đang chọn"):
            st.dataframe(pd.DataFrame({"Biến": NUMERIC_FEATURES,
                                      "Giá trị": [record[name] for name in NUMERIC_FEATURES],
                                      "Đơn vị": [UNITS[name] for name in NUMERIC_FEATURES]}),
                         hide_index=True, width="stretch")
        submitted = st.button("Dự đoán giờ tiếp theo", type="primary", key="predict_sample")
    else:
        # Widget ngoài form gửi thay đổi ngay, để ẩn dự đoán của đầu vào cũ.
        with st.container():
            date_col, hour_col = st.columns(2)
            date = date_col.date_input("Ngày quan trắc", value=default.datetime.date())
            hour = hour_col.number_input("Giờ quan trắc (0–23)", min_value=0, max_value=23,
                                         value=int(default.datetime.hour), step=1)
            record = {"station": station, "datetime": datetime.combine(date, datetime.min.time()).replace(hour=int(hour))}
            st.caption("Các chỉ số hiện tại")
            columns = st.columns(3)
            for index, name in enumerate(NUMERIC_FEATURES):
                minimum = None if name in {"TEMP", "DEWP"} else (0.01 if name == "PRES" else 0.0)
                record[name] = columns[index % 3].number_input(
                    LABELS[name], value=float(default[name]), min_value=minimum,
                    key=f"manual_{station}_{name}", format="%.2f",
                )
            st.caption("Điểm sương DEWP khác độ ẩm tương đối. Ba giá trị trễ phải đúng t−1, t−2, t−3.")
            submitted = st.button("Dự đoán giờ tiếp theo", type="primary", key="predict_manual")
    signature = json.dumps({name: str(record[name]) for name in ["station", "datetime", *NUMERIC_FEATURES]}, sort_keys=True)
    signature += str(version)
    result_key = "module1_prediction"
    if submitted:
        try:
            value = predict_pm25(record, model_path=output / "linear_regression.joblib")
            st.session_state[result_key] = {"signature": signature, "value": value}
        except (ValueError, OSError) as exc:
            st.session_state.pop(result_key, None)
            st.error(str(exc))
    result = st.session_state.get(result_key)
    if result and result["signature"] == signature:
        target_time = pd.Timestamp(record["datetime"]) + pd.Timedelta(hours=1)
        st.markdown(f"**Kết quả cho {station} lúc {target_time:%d/%m/%Y %H:%M}**")
        columns = st.columns(3)
        columns[0].metric("PM2.5 dự đoán", f"{result['value']:.2f} µg/m³")
        columns[1].metric("Đối chứng persistence", f"{record['PM2.5']:.2f} µg/m³")
        if mode == "Chọn mẫu test":
            columns[2].metric("PM2.5 thực tế", f"{record['actual']:.2f} µg/m³")
            st.caption(f"Sai số tuyệt đối của dự đoán này: {abs(result['value'] - record['actual']):.2f} µg/m³.")
        if result["value"] < 0:
            st.warning("Mô hình tuyến tính cho kết quả âm ở đầu vào này. Kết quả được giữ nguyên; nồng độ thực tế không âm.")
        exported = {**record, "target_datetime": target_time, "predicted_pm25": result["value"]}
        st.download_button("Tải kết quả CSV", pd.DataFrame([exported]).to_csv(index=False).encode("utf-8-sig"),
                           file_name="module1_prediction.csv", mime="text/csv", key="download_prediction")
    elif result:
        st.info("Đầu vào hoặc mô hình đã thay đổi. Bấm dự đoán để cập nhật kết quả.")


def render_evaluation(metrics, predictions, coefficients):
    st.subheader("Đánh giá và đối chứng")
    split = st.selectbox("Tập đánh giá", ["test", "validation"], key="evaluation_split")
    scores = metrics[metrics.split.eq(split)].set_index("model")
    lr, baseline = scores.loc["Linear Regression"], scores.loc["Persistence"]
    columns = st.columns(3)
    for column, metric in zip(columns, ["MAE", "RMSE", "R2"]):
        column.metric(metric if metric != "R2" else "R²", f"{lr[metric]:.4f}",
                      delta=f"{lr[metric]-baseline[metric]:+.4f} so với persistence",
                      delta_color="normal" if metric == "R2" else "inverse")
    st.caption("MAE/RMSE: µg/m³, càng thấp càng tốt. R²: càng cao càng tốt. Persistence lấy PM2.5 hiện tại làm dự đoán giờ sau.")
    st.dataframe(scores.reset_index(), hide_index=True, width="stretch")
    station = st.selectbox("Trạm trên biểu đồ", sorted(predictions.station.unique()), key="evaluation_station")
    subset = predictions[predictions.split.eq(split) & predictions.station.eq(station)].sort_values("target_datetime")
    count = st.slider("Số mẫu đầu hiển thị", 24, 336, 168, 24)
    plot = subset.head(count).set_index("target_datetime")
    st.line_chart(plot[["actual", "predicted", "baseline"]].rename(
        columns={"actual": "Thực tế", "predicted": "Linear Regression", "baseline": "Persistence"}),
        y_label="PM2.5 (µg/m³)")
    st.caption("Biểu đồ dùng giờ cần dự đoán. Chỉ số phía trên tính trên toàn bộ trạm trong tập được chọn.")
    st.line_chart(plot[["error"]].rename(columns={"error": "Dự đoán − thực tế"}), y_label="Sai số (µg/m³)")
    st.download_button("Tải dự đoán của trạm", subset.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"module1_{station}_{split}.csv", mime="text/csv")
    st.subheader("Hệ số hồi quy")
    numeric = coefficients[coefficients.kind.eq("numeric")].set_index("feature")
    st.bar_chart(numeric[["coefficient"]].rename(columns={"coefficient": "Hệ số chuẩn hóa"}), horizontal=True)
    st.caption("Biến số tăng 1 độ lệch chuẩn train, giữ các đầu vào khác cố định. Các hệ số không chứng minh quan hệ nhân quả.")
    with st.expander("Bảng hệ số đầy đủ"):
        st.dataframe(coefficients, hide_index=True, width="stretch")


def render(output=OUTPUT_DIR):
    output = Path(output)
    st.title("Dự báo PM2.5 theo giờ")
    st.caption("Module 1 · Linear Regression · Dữ liệu 12 trạm Bắc Kinh")
    required = ["linear_regression.joblib", "metrics.csv", "predictions.csv", "coefficients.csv"]
    if (output / ".training").exists() or any(not (output / name).is_file() for name in required):
        st.info("Kết quả chưa sẵn sàng. Chạy python run_module1.py --train-only rồi mở lại trang.")
        return
    version = (output / "linear_regression.joblib").stat().st_mtime_ns
    try:
        metadata, metrics, predictions, samples, coefficients = load_results(str(output.resolve()), version)
    except (OSError, ValueError) as exc:
        st.error(f"Không đọc được kết quả: {exc}. Hãy huấn luyện lại.")
        return
    prediction_tab, evaluation_tab, method_tab = st.tabs(["Dự đoán", "Đánh giá", "Phương pháp"])
    with prediction_tab:
        render_prediction(output, metadata, samples, version)
    with evaluation_tab:
        render_evaluation(metrics, predictions, coefficients)
    with method_tab:
        st.subheader("Từ dữ liệu đến dự đoán")
        st.markdown("Quan trắc đến **t** → ghép PM2.5 tại **t−1, t−2, t−3** → tạo biến lịch → chuẩn hóa/mã hóa → dự đoán **PM2.5(t+1)**.")
        st.latex(r"\hat{y}=\beta_0+\sum_{i=1}^{n}\beta_i x_i")
        st.write("Chỉ fit trên train. Ghép theo đúng trạm và giờ; không điền thiếu từ tương lai. Nhãn là giá trị thực đúng một giờ sau.")
        st.dataframe(pd.DataFrame({"Tập": ["Train", "Validation", "Test"],
                                  "Thời điểm nhãn": ["Trước 01/11/2015", "01/11/2015 – trước 01/07/2016", "Từ 01/07/2016"],
                                  "Số mẫu": list(metadata["split_rows"].values())}),
                     hide_index=True, width="stretch")
        st.write("Dự báo cuốn từng giờ: mỗi lần dự đoán đều có quan trắc thực mới đến t. Các chỉ số không mô tả dự báo nhiều bước liên tục.")
        st.write("Loại mẫu thiếu có thể gây thiên lệch. Mô hình tuyến tính có thể bỏ lỡ đột biến và cho dự đoán âm. Dữ liệu Bắc Kinh chưa đại diện cho Việt Nam.")
        st.caption("Đây là nồng độ PM2.5 theo giờ, không phải chỉ số AQI hay đánh giá sức khỏe.")
        st.markdown(f"[Nguồn dữ liệu UCI]({metadata['source']})")


if __name__ == "__main__":
    st.set_page_config(page_title="Module 1 · Dự báo PM2.5", layout="wide")
    render()
