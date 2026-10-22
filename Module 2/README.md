# Module 2 Logistic Regression

Dự báo nhóm PM2.5 giờ kế tiếp tại 12 trạm Bắc Kinh. Nhãn Thấp / Trung bình / Cao dùng phân vị 1/3 và 2/3 của mục tiêu train, **không phải AQI chính thức**. Xem kết quả đo thực tế trong [reports/REPORT.md](reports/REPORT.md).

## Chạy

Mở terminal tại thư mục `Module 2`, dùng Python 3.11 trở lên:

```powershell
python -m pip install -r requirements.txt
python train.py
python -m unittest -v test_module2.py test_app.py
python verify_notebook.py
python -m streamlit run app.py --server.port 8503
```

Mô hình đã huấn luyện nằm trong `artifacts/`; có thể mở demo ngay nếu môi trường đã có thư viện. `train.py` đọc 12 file `../data/PRSA_Data_*.csv`, không sửa dữ liệu gốc. Hai file dữ liệu chứng khoán được bỏ qua. Chạy lại ghi đè các kết quả sinh tự động trong artifacts và reports.

## Dashboard demo

Tại thư mục `ML`, vào module bằng `cd "Module 2"` (cần ngoặc kép vì tên có dấu cách). Chạy `python -m streamlit run app.py --server.port 8503`, mở http://localhost:8503. Nếu ứng dụng đang chạy, tải lại trang để nhận giao diện mới; không cần huấn luyện lại.

Điều hướng bên trái gồm sáu mục:

- **Tổng quan dự án:** số liệu chính, quy trình dự báo, so sánh baseline và vai trò của các module.
- **Khám phá dữ liệu:** lọc nhiều trạm, giai đoạn và biến; xem chuỗi thời gian, phân phối, chu kỳ giờ, tỷ lệ thiếu và tải CSV.
- **Nhãn & mô hình:** ngưỡng nhãn, số mẫu/chia tập, công thức, pipeline, schema và hệ số theo lớp.
- **Thực nghiệm & đánh giá:** chọn test/validation, so sánh trên toàn bộ hoặc tập con có PM2.5(t), confusion matrix, báo cáo từng lớp, các lượt tuning, lát cắt trạm/tháng/giờ và tra cứu lỗi.
- **Dự đoán tương tác:** chọn trạm/ngày/giờ, xem nhãn/xác suất, 48 giờ lịch sử, đặc trưng đầu vào, đối chiếu thực tế và baseline, tải JSON. Khi đổi đầu vào, kết quả cũ được ẩn cho đến khi dự đoán lại.
- **Báo cáo & bàn giao:** báo cáo kết quả, tải notebook/metrics/metadata và hướng dẫn tích hợp.

Luồng thuyết trình gợi ý: Tổng quan → Dữ liệu → Nhãn & mô hình → Đánh giá → Dự đoán → Bàn giao. Biểu đồ EDA mặc định dùng train; bộ lọc EDA theo thời gian quan trắc, còn train/validation/test của mô hình theo thời gian mục tiêu. Confusion matrix luôn hiển thị toàn bộ tập được chọn và có chú thích rõ.

Các bảng hiển thị đầy đủ hàng theo bộ lọc, không cắt ở 300/500 hàng. Bảng ngắn mở hết chiều cao; bảng dài dùng vùng cuộn riêng. Đặt con trỏ trong bảng để cuộn dữ liệu, kéo thanh cuộn ngang ở đáy bảng để xem cột bên phải; đưa con trỏ ra ngoài để cuộn trang. Nút phóng to của bảng giúp xem trên màn hình nhỏ. Chọn nhiều trạm và toàn bộ giai đoạn sẽ cần tải nhiều dữ liệu hơn.

Mở `module2.ipynb` để xem EDA, thực nghiệm, báo cáo và minh họa API. Notebook dùng chính mã của train và app; đặt `RUN_TRAINING = True` để huấn luyện lại từ notebook.

## Thiết kế

`module2.py` chứa đọc/kiểm tra dữ liệu, lưới giờ, feature builder, nhãn, pipeline và API. `train.py` thực hiện chọn đặc trưng/tham số trên validation, đánh giá test, xuất mô hình và báo cáo. `app.py` là demo lịch sử, `test_module2.py` kiểm tra các ràng buộc quan trọng.

Train có target_time trước 01/03/2015; validation từ 01/03/2015 đến trước 01/03/2016; test từ 01/03/2016 đến trước 01/03/2017. Ngưỡng nhãn và preprocessing chỉ fit trên train. Không refit sau khi chọn cấu hình. Dự báo cuốn chiếu một giờ: mỗi lần được dùng số đo thật đến t.

Số đo không hợp lệ chuyển thành thiếu. Không điền mục tiêu. Trùng hoàn toàn được bỏ; trùng khóa nhưng mâu thuẫn gây lỗi. Lưới giờ được bổ sung riêng từng trạm. Thiếu đặc trưng dùng median train và cờ thiếu; biến phân loại dùng giá trị `missing`. Rolling kết thúc tại t, min_periods=1, std ddof=1, kèm số quan sát.

## Tích hợp

```python
from module2 import load_data, build_features, predict_pollution_class
history, audit, missing = load_data()
features = build_features(history, 'Aotizhongxin', '2016-05-01 09:00')
result = predict_pollution_class(features)
print(result)
```

Đầu vào API là một hàng đặc trưng theo `artifacts/metadata.json`, có `timestamp`. Thiếu cột bị từ chối; giá trị NaN được impute và báo số đặc trưng thiếu. Trạm mới bị từ chối. Timestamp không timezone được hiểu là giờ Bắc Kinh; timestamp có timezone được đổi sang Asia/Shanghai. Phải là giờ tròn. Đơn vị nồng độ µg/m³, TEMP/DEWP °C, PRES hPa, RAIN mm, WSPM m/s. Chỉ tải joblib do bạn tin cậy.

`build_features` nhận lịch sử đã làm sạch có timestamp dạng datetime không timezone theo giờ Bắc Kinh (đầu ra `load_data`). Hàm giới hạn dữ liệu đến t và tái tạo đúng 24 giờ lịch sử; không dùng số đo tương lai.

## Kết quả bàn giao

- `artifacts/model.joblib`, `metadata.json`, `label_policy.json`: pipeline, schema, ngưỡng, checksum và phiên bản.
- `reports/data_audit.json`, `missing_by_station.csv`, `eda.png`: kiểm tra dữ liệu và EDA train.
- `reports/experiments.csv`, `metrics.csv`, `classification_report.json`: thực nghiệm, baseline, precision/recall/F1/support.
- `reports/confusion_*.png`, `test_slices.csv`, `test_predictions.csv`: lỗi theo lớp, trạm/tháng/giờ và dự đoán từng mẫu.
- `reports/coefficients.csv`, `coefficients.png`, `prediction_example.json`: diễn giải và ví dụ tích hợp.

Nguồn: Chen, S. (2017), UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G, CC BY 4.0. Dữ liệu có sẵn trong dự án; không biết thời điểm tải gốc. Không diễn giải hệ số thành nhân quả hoặc kết quả Bắc Kinh thành độ chính xác tại Việt Nam. Xác suất chưa được hiệu chỉnh.
