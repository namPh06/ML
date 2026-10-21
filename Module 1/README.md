# Module 1 — Linear Regression dự đoán PM2.5 giờ tiếp theo

**Một luồng chạy:** `run_module1.py → module1.ipynb → outputs → app.py`.

## 1. Cài và chạy

Cần Python 3.11 trở lên, khuyên dùng Python 3.12. Dữ liệu đã có trong `ML/data`, không cần tải thêm.

Mở terminal tại thư mục này:

```powershell
cd "G:\nam4ki1\hocmay\btl\ML\Module 1"
python -m pip install -r requirements.txt
python run_module1.py
```

Chương trình sẽ:
1. Chạy tất cả ô code trong **module1.ipynb** bằng đúng Python hiện tại.
2. Lưu bảng/biểu đồ mới vào chính notebook và ghi kết quả vào **outputs**.
3. Mở ứng dụng tại **http://127.0.0.1:8502**. Truy cập địa chỉ này để dùng.

Chạy lại lệnh sẽ huấn luyện lại. Nhấn **Ctrl+C** trong terminal để dừng ứng dụng.
Nếu cổng bận, dùng `python run_module1.py --port 8503`.
Nếu chỉ cần chạy notebook và xuất kết quả: `python run_module1.py --train-only`.
Nếu notebook có lỗi, chương trình dừng trước khi mở ứng dụng; mở notebook để xem ô bị lỗi.

## 2. Đọc file nào trước?

```text
Module 1/
├── module1.ipynb          Toàn bộ bài làm và huấn luyện
├── app.py                 Giao diện và hàm predict_pm25
├── run_module1.py         Chạy notebook rồi mở ứng dụng
├── requirements.txt      Một danh sách thư viện
├── README.md              Hướng dẫn này
├── .gitignore             Bỏ qua file tạm và dữ liệu sinh lại được
└── outputs/
    ├── linear_regression.joblib
    ├── metrics.csv
    ├── predictions.csv
    ├── coefficients.csv
    ├── actual_vs_predicted.png
    └── error_over_time.png
```

**Học và báo cáo:** đọc notebook từ trên xuống, chạy từng ô để xem kết quả.
Có thể mở notebook bằng trình soạn thảo hỗ trợ Jupyter hoặc lệnh:

```powershell
python -m notebook module1.ipynb
```

Notebook chứa trực tiếp code đọc dữ liệu, tạo đặc trưng, chia tập, huấn luyện và đánh giá.
Sửa cách huấn luyện trong notebook thì lần chạy `run_module1.py` tiếp theo sẽ dùng đúng phần vừa sửa.

**Ứng dụng:** đọc mô hình đã lưu. Bấm dự đoán chỉ sử dụng mô hình, không huấn luyện lại.
`app.py` có hàm `render()` để gọi khi ghép vào giao diện nhóm.

## 3. Bài toán và phương pháp

- Đọc 12 file `PRSA_Data_*.csv` trong `../data`; mỗi file là một trạm Bắc Kinh.
- Sáu biến hiện tại: PM2.5, PM10, TEMP, PRES, DEWP, WSPM.
- Thêm PM2.5 đúng t−1, t−2, t−3, cùng trạm; nhãn là PM2.5 đúng t+1.
- Thêm trạm, giờ, thứ trong tuần, tháng; không dùng quan trắc tương lai làm đầu vào.
- Không nội suy. Loại mẫu thiếu sau khi tra cứu đúng giờ; trùng trạm/giờ xung đột thì báo lỗi.
- Chia theo **giờ của nhãn**: train trước 01/11/2015; validation trước 01/07/2016; test từ 01/07/2016.
- Pipeline: StandardScaler cho biến số → OneHotEncoder cho biến phân loại → LinearRegression.
- Chỉ fit trên train. Validation/test chỉ dùng dự đoán và đánh giá; cấu hình không được chọn bằng test.
- Đối chứng persistence: dự đoán giờ sau bằng PM2.5 hiện tại.
- Đánh giá MAE, RMSE, R². MAE/RMSE càng thấp càng tốt; R² càng cao càng tốt.

DEWP là nhiệt độ điểm sương, không phải độ ẩm tương đối.
Thời gian là giờ Bắc Kinh UTC+8, lưu không gắn timezone.
Đây là dự báo cuốn từng giờ với quan trắc thực mới đến t, không phải dự báo nhiều bước liên tục.

## 4. Các kết quả trong outputs

| File | Nội dung |
|---|---|
| linear_regression.joblib | Pipeline đã học và thông tin đầu vào để ứng dụng dùng lại. |
| metrics.csv | MAE, RMSE, R² của Linear Regression và persistence trên ba tập. |
| predictions.csv | Đầu vào, giờ nguồn/giờ nhãn, thực tế, dự đoán và sai số của validation/test. |
| coefficients.csv | Hệ số chuẩn hóa, hệ số theo đơn vị gốc và mức phân loại. |
| actual_vs_predicted.png | So sánh thực tế/dự đoán theo giờ và biểu đồ phân tán. |
| error_over_time.png | Sai số theo giờ và sai số trung bình ngày của một trạm test. |

Các file này do notebook tạo, không cần sửa bằng tay.
Biểu đồ hệ số và bảng kiểm tra dữ liệu nằm ngay trong notebook để dùng khi báo cáo.
Mô hình và predictions.csv được bỏ qua khi dùng Git vì có thể sinh lại bằng lệnh chạy.

## 5. Hàm dự đoán để tích hợp nhóm

Khi đã chạy notebook, có thể gọi từ thư mục Module 1:

```python
from app import predict_pm25

value = predict_pm25({
    "station": "Aotizhongxin",
    "datetime": "2016-07-01 09:00",
    "PM2.5": 75, "PM10": 100,
    "TEMP": 28, "PRES": 1005, "DEWP": 21, "WSPM": 2,
    "PM25_lag_1": 90, "PM25_lag_2": 100, "PM25_lag_3": 105,
})
print(value)  # PM2.5 dự đoán cho 10:00, đơn vị µg/m³.
```

Đầu vào phải đủ chín biến số, tên trạm và giờ tròn. Không chấp nhận giá trị thiếu/vô hạn,
trạm chưa học hoặc thời gian có timezone. Nhiệt độ và điểm sương có thể âm; PM và tốc độ gió không âm; áp suất phải dương.

## 6. Chuẩn bị báo cáo

Trình bày theo thứ tự: bài toán → dữ liệu → đặc trưng/nhãn → chia thời gian → công thức/pipeline
→ MAE/RMSE/R² và persistence → hệ số/biểu đồ → demo → hạn chế.

Hệ số biến số ứng với một độ lệch chuẩn train; cột coefficient_per_unit đổi về một đơn vị gốc.
Hệ số phân loại so với mức tham chiếu. Không diễn giải thành quan hệ nhân quả.

Hạn chế: mẫu bị loại có thể gây thiên lệch; mô hình có thể bỏ lỡ đột biến và cho dự đoán âm;
dữ liệu Bắc Kinh chưa đại diện cho Việt Nam. Giữ nguyên dự đoán âm để kết quả đánh giá và giao diện nhất quán.
Dự đoán nồng độ PM2.5 không tương đương chỉ số AQI hoặc đánh giá sức khỏe.
