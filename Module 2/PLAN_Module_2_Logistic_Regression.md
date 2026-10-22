# Kế hoạch triển khai Module 2 — Logistic Regression

**Đề tài:** Dự đoán chất lượng không khí  
**Dataset:** Beijing Multi-Site Air Quality — UCI, ID 501  
**Ngày lập:** 21/10/2026  
**Phạm vi:** Kế hoạch thực hiện, chưa phải báo cáo kết quả huấn luyện.

> Tài liệu dựa trên file `De_tai_du_doan_chat_luong_khong_khi.docx` và metadata chính thức của UCI. Chưa nhận bộ CSV thực tế từ người dùng, chưa chạy EDA hoặc huấn luyện. Các mốc thời gian, cấu hình và tiêu chí dưới đây là phương án triển khai đề xuất; số dòng sau làm sạch và các chỉ số phải được đo khi thực hiện.

## 1. Mục tiêu và quyết định triển khai

Tại một trạm và thời điểm t, dùng những thông tin đã có đến t để dự đoán nhóm nồng độ PM2.5 tại t+1 giờ. Mô hình phải trả về nhãn và xác suất từng lớp.

| Hạng mục | Phương án mặc định |
|---|---|
| Mô hình chính | Logistic Regression đa lớp |
| Chân trời dự báo | 1 giờ |
| Phạm vi thử nhanh | Một trạm: Aotizhongxin, nếu có trong bản tải |
| Phạm vi nghiệm thu | Toàn bộ 12 trạm, một mô hình chung có mã hóa station |
| Nhãn MVP | Thấp / Trung bình / Cao theo phân vị trên train, không phải AQI |
| Chỉ số chọn mô hình | Macro-F1 trên validation |
| Baseline | Lớp phổ biến nhất và giữ nguyên nhóm PM2.5 hiện tại |
| Demo | Chọn trạm và thời điểm lịch sử trên Streamlit |
| Lưu mô hình | Pipeline + cấu hình đặc trưng + ngưỡng nhãn + metadata |

**Cổng quyết định cần thống nhất với nhóm/giảng viên:** Nếu bắt buộc phải dùng tiêu chuẩn AQI chính thức thì hoàn thành mục 3.2 trước khi tạo nhãn cuối cùng. Trong lúc chưa chốt, có thể phát triển pipeline bằng nhãn phân vị, nhưng không trình bày nó như AQI hoặc các mức an toàn sức khỏe.

### Kết quả mong đợi

- Một pipeline tái chạy được từ CSV gốc đến báo cáo đánh giá.
- Chứng minh đúng thời điểm dự báo và không rò rỉ dữ liệu.
- So sánh công bằng với baseline trên cùng các hàng dữ liệu.
- Có hàm tích hợp `predict_pollution_class(input_data)`.
- Nếu mô hình không vượt baseline, báo cáo trung thực nguyên nhân và hạn chế; không đặt trước Accuracy cần đạt.

## 2. Hiểu và kiểm tra dataset

Theo UCI [S1], bộ dữ liệu có 420.768 bản ghi theo giờ, 12 trạm, từ 01/03/2013 đến 28/02/2017; có dữ liệu thiếu ký hiệu NA. Metadata nêu các chất ô nhiễm và biến khí tượng sau:

| Cột | Ý nghĩa / đơn vị | Vai trò dự kiến |
|---|---|---|
| No | Số thứ tự | Loại khỏi đặc trưng |
| year, month, day, hour | Thành phần thời gian | Ghép timestamp |
| PM2.5, PM10, SO2, NO2, CO, O3 | Nồng độ, µg/m³ | Đặc trưng tại t; PM2.5 tương lai tạo nhãn |
| TEMP | Nhiệt độ, °C | Đặc trưng |
| PRES | Áp suất, hPa | Đặc trưng |
| DEWP | Nhiệt độ điểm sương, °C | Đặc trưng |
| RAIN | Lượng mưa, mm | Đặc trưng |
| wd | Hướng gió | Biến phân loại |
| WSPM | Tốc độ gió, m/s | Đặc trưng |
| station | Tên trạm | Khóa nhóm và biến phân loại |

Không có sẵn cột độ ẩm tương đối trong danh sách này. MVP dùng DEWP; không tự đổi tên DEWP thành humidity. Nếu bổ sung độ ẩm tính toán sau này, phải ghi công thức và kiểm chứng riêng.

### Task 2.1 — Tải và lưu nguồn

- [ ] Tải gói `PRSA2017_Data_20130301-20170228.zip` từ trang UCI [S1].
- [ ] Giải nén, chọn các CSV trạm tương ứng; không vô tình dùng CSV mẫu/test khác trên trang như toàn bộ dataset.
- [ ] Giữ nguyên bản raw, ghi URL, thời điểm tải và SHA-256 các file.
- [ ] Ghi công nhận nguồn Chen, S. (2017), DOI: 10.24432/C5RK5G và giấy phép CC BY 4.0.

### Task 2.2 — Kiểm tra schema và EDA

- [ ] Kiểm tra cột, kiểu dữ liệu và đơn vị trước khi nối file.
- [ ] Ghép timestamp; coi giờ trong dataset là giờ địa phương Bắc Kinh, ghi giả định này trong metadata.
- [ ] Đếm số trạm, số hàng, phạm vi thời gian, khóa trùng `(station, timestamp)`.
- [ ] Đếm thiếu theo cột, trạm; kiểm tra khoảng giờ bị khuyết.
- [ ] Kiểm tra giá trị không hợp lệ, ví dụ nồng độ âm; nhiệt độ âm vẫn có thể hợp lệ.
- [ ] Vẽ phân phối PM2.5 và chuỗi thời gian trên train; ghi nhận đuôi dài và biến động.
- [ ] Không loại mọi nồng độ cao như ngoại lệ: đây có thể là tình huống cần mô hình nhận diện.

**Đầu ra:** `data_audit.json`, bảng thiếu dữ liệu, biểu đồ EDA. Kiểm tra schema toàn bộ dữ liệu được phép; lựa chọn ngưỡng/đặc trưng dựa trên train và validation, không dựa trên hiệu năng test.

## 3. Thiết kế nhãn

### 3.1. Nhãn MVP chạy được ngay: ba nhóm nồng độ tương đối

Đây là lựa chọn kỹ thuật của kế hoạch, không phải ngưỡng quy định. Tính q1 và q2 là phân vị 1/3 và 2/3 của **PM2.5 mục tiêu quan sát được trong train**, gộp các trạm của phạm vi huấn luyện. Không tính riêng mỗi trạm vì sẽ làm cùng một nồng độ mang nhãn khác nhau.

| Mã | Tên | Quy tắc với nồng độ p |
|---|---|---|
| 0 | Thấp | 0 ≤ p ≤ q1 |
| 1 | Trung bình | q1 < p ≤ q2 |
| 2 | Cao | p > q2 |

- Lưu q1, q2 thành số thực đầy đủ, không làm tròn trước khi gán nhãn.
- Nếu q1 ≥ q2, dừng và xem lại dữ liệu; không tự tạo lớp rỗng.
- NaN hoặc giá trị không hợp lệ không được gán lớp.
- Dùng đúng ngưỡng train cho validation, test và demo.
- Đóng băng ngưỡng kể cả khi refit train+validation để kết quả còn so sánh được.
- Nhãn này mô tả mức tương đối trong bộ dữ liệu; không suy ra tác động sức khỏe hay khẳng định không khí an toàn.
- Phân vị không đảm bảo số mẫu bằng nhau hoàn toàn do giá trị trùng; vẫn phải thống kê support.

**Đầu ra:** `label_policy.json` gồm strategy, ngưỡng, tên lớp, khoảng biên, đơn vị, thời gian fit và version.

### 3.2. Nếu đề tài yêu cầu ngưỡng AQI chính thức

Thực hiện trước khi huấn luyện bản nghiệm thu:

1. Chọn quốc gia/tiêu chuẩn, phiên bản và tài liệu gốc.
2. Xác minh quy tắc trung bình thời gian, làm tròn, xử lý thiếu và khoảng nồng độ.
3. Xác định đang dự báo chỉ số thành phần PM2.5 hay AQI tổng hợp nhiều chất.
4. Nếu tiêu chuẩn yêu cầu trung bình nhiều giờ, sửa định nghĩa mục tiêu theo đúng cửa sổ kết thúc tại t+1; không dùng trực tiếp một số đo giờ rồi gọi là AQI.
5. Ghi rõ mục tiêu mới; chia tập theo thời điểm cuối của thông tin dùng trong nhãn.
6. Dùng tiêu chuẩn đó thống nhất trong các module có hiển thị nhóm ô nhiễm.

Không tự dùng ba tên Tốt/Trung bình/Kém cùng các ngưỡng tùy ý. Nếu thay đổi nhãn sau MVP, phải huấn luyện và đánh giá lại, không so trực tiếp các F1 của hai cách định nghĩa khác nhau.

## 4. Xây dựng bảng thời gian và mục tiêu

### 4.1. Làm sạch tối thiểu

1. Nối CSV, chuẩn hóa tên cột theo bảng ánh xạ đã lưu.
2. Xử lý bản ghi trùng: bản giống hệt có thể bỏ; bản mâu thuẫn phải báo cáo và có quy tắc rõ ràng.
3. Sắp xếp theo station và timestamp.
4. Tạo lưới giờ đều riêng mỗi trạm; hàng thêm vào mang NaN và có cờ nhận diện.
5. Giữ riêng `pm25_observed` từ số đo hợp lệ, không điền thiếu vào cột này.

### 4.2. Tạo mục tiêu

Với mỗi hàng tại t:

```python
# Chỉ dùng sau khi đã có lưới giờ đều và sort trong mỗi trạm.
df['target_time'] = df['timestamp'] + pd.Timedelta(hours=1)
df['pm25_target'] = df.groupby('station')['pm25_observed'].shift(-1)
```

Kiểm tra bằng tay ít nhất 5 mẫu, gồm điểm chuyển ngày và ranh giới giữa trạm. Có thể dùng self-join theo `(station, timestamp + 1h)` để lấy mục tiêu nếu không reindex.

- Bỏ mẫu có mục tiêu thiếu hoặc không hợp lệ.
- Không nội suy, forward-fill hoặc dùng median để tạo đáp án.
- `target_time`, `pm25_target` và nhãn chỉ thuộc metadata/y, tuyệt đối không thuộc X.

## 5. Đặc trưng và chống rò rỉ

### 5.1. Bộ đặc trưng đề xuất

| Nhóm | Đặc trưng | Ưu tiên |
|---|---|---|
| Hiện tại | PM2.5, PM10, SO2, NO2, CO, O3 tại t | Bắt buộc |
| Thời tiết | TEMP, PRES, DEWP, RAIN, WSPM tại t | Bắt buộc |
| Phân loại | wd, station | Bắt buộc |
| Lịch sử | PM2.5 tại t−1, t−2, t−3 | Bắt buộc |
| Chu kỳ | sin/cos của hour và month; day_of_week | Bắt buộc |
| Mở rộng | PM2.5 t−6, t−12, t−24 | Sau MVP |
| Cửa sổ | Mean/std 3 giờ và 6 giờ kết thúc tại t | Sau MVP |
| Xu hướng | PM2.5(t) − PM2.5(t−1) | Sau MVP |

Tạo hour_sin = sin(2π·hour/24), hour_cos = cos(2π·hour/24); month dùng (month−1)/12. Không dùng No làm tín hiệu dự đoán.

MVP giữ dữ liệu thiếu trong đặc trưng và dùng imputer trong pipeline. Tạo lag từ chuỗi quan sát trên lưới giờ, không từ mục tiêu tương lai. Nếu bổ sung rolling, quy định rõ min_periods, ddof và cờ số quan sát; tuyệt đối không dùng cửa sổ centered. Với cửa sổ có chứa t, giả định số đo giờ t đã sẵn sàng khi dự báo.

### 5.2. Các nguyên tắc bắt buộc

- groupby station trước shift/rolling; không trộn lịch sử hai trạm.
- Không backward-fill hoặc nội suy hai chiều dùng tương lai.
- Imputer/scaler/encoder chỉ fit trên train của từng lần huấn luyện.
- Nếu thử forward-fill, giới hạn khoảng thiếu và làm nhân quả riêng từng trạm; phục vụ dự đoán phải tái hiện được quy tắc đó.
- Không fit preprocessing trên cả dataset rồi mới chia tập.
- Lịch sử ở cuối train có thể được dùng làm đầu vào validation; đây là dữ liệu đã biết tại thời điểm dự báo.
- Đánh giá theo cơ chế dự báo cuốn chiếu một giờ: ở mỗi t trong test, số đo thực tế đến t được phép dùng. Không tuyên bố đây là dự báo toàn bộ test từ một mốc ban đầu.

## 6. Chia tập theo thời gian

Chia theo **target_time** để nhãn train không tràn sang giai đoạn validation. Áp dụng chung mốc cho mọi trạm.

| Tập | Khoảng target_time, đầu gồm/cuối không gồm | Mục đích |
|---|---|---|
| Train | [2013-03-01, 2015-03-01) | Fit nhãn phân vị, preprocessing và mô hình |
| Validation | [2015-03-01, 2016-03-01) | Chọn đặc trưng và tham số |
| Test | [2016-03-01, 2017-03-01) | Đánh giá cuối cùng |

Mốc này là đề xuất giúp mỗi giai đoạn đánh giá bao phủ các mùa. Số mẫu đầu kỳ giảm do chưa đủ lịch sử hoặc không có mục tiêu. Ghi số hàng thực tế sau lọc cho từng trạm/tập.

Không train_test_split ngẫu nhiên; không dùng KFold trộn dữ liệu. Nếu bổ sung cross-validation, chia theo các timestamp duy nhất, cho tất cả trạm cùng một thời điểm vào cùng fold. Không áp dụng TimeSeriesSplit trực tiếp lên bảng đang xếp từng trạm nối nhau. Với nhãn phân vị, mỗi fold fit ngưỡng từ train fold và đánh giá theo chính ngưỡng đó; ghi sự khác biệt của các ngưỡng.

Sau khi chọn cấu hình, có thể refit preprocessing và mô hình trên train+validation, giữ ngưỡng nhãn đã đóng băng từ train. Chỉ dùng test khi cấu hình cuối đã được khóa.

## 7. Tiền xử lý và mô hình

### 7.1. Pipeline dự kiến

- Numeric: SimpleImputer(strategy='median', add_indicator=True) → StandardScaler.
- Categorical: điền hằng 'missing' → OneHotEncoder(handle_unknown='ignore').
- Kết hợp bằng ColumnTransformer → LogisticRegression.
- Kiểm tra không có cột số hoàn toàn thiếu trong train; nếu có thì loại bằng quy tắc ghi trong schema hoặc xử lý cố định có chủ đích.
- API cụ thể cần đối chiếu phiên bản cài đặt [S2]; ghi phiên bản vào requirements/metadata.

### 7.2. Kiến thức cần giải thích khi bảo vệ

Với K lớp, xác suất dạng softmax:

\[
P(y=k\mid x)=\frac{\exp(w_k^Tx+b_k)}{\sum_{j=1}^{K}\exp(w_j^Tx+b_j)}.
\]

Lớp dự đoán là lớp có xác suất lớn nhất. Mô hình tối ưu hàm mất mát phân loại có điều chuẩn. C điều khiển nghịch mức điều chuẩn: C nhỏ hơn tương ứng điều chuẩn mạnh hơn. Chuẩn hóa giúp các biến khác thang đo được xử lý nhất quán. Logistic Regression mặc định không trực tiếp mô hình hóa thứ tự Thấp–Trung bình–Cao; cần quan sát lỗi nhầm xa giữa hai đầu.

### 7.3. Lưới thực nghiệm vừa đủ

| Thành phần | Giá trị đề xuất |
|---|---|
| solver | lbfgs |
| C | 0.01, 0.1, 1, 10 |
| class_weight | None, balanced |
| max_iter | Bắt đầu 2000; tăng nếu chưa hội tụ |
| Tập đặc trưng | Cơ bản; cơ bản + lag; thêm rolling/xu hướng |

Huấn luyện từ cấu hình C=1, class_weight=None và bộ cơ bản + lag. Không chạy toàn bộ tổ hợp ngay. Chọn bộ đặc trưng trên validation trước, sau đó thử tối đa 8 tổ hợp C/weight. Ghi warning hội tụ, thời gian fit và metric; không ẩn warning rồi coi là hoàn tất.

Xác suất dự đoán là xác suất mô hình, không phải mức chắc chắn đã được kiểm chứng. Chỉ thêm calibration nếu thực sự cần, dùng dữ liệu huấn luyện/validation riêng theo thời gian, không calibrate bằng test.

## 8. Baseline và đánh giá

### 8.1. Hai baseline bắt buộc

1. Majority: luôn dự đoán lớp nhiều nhất trong train.
2. Persistence: dùng cùng label_policy để phân nhóm PM2.5 tại t làm dự đoán cho t+1.

Nếu PM2.5 tại t thiếu, baseline persistence dùng fallback lớp phổ biến nhất của train và ghi tỷ lệ fallback. Đồng thời báo cáo thêm so sánh trên tập con có PM2.5(t) quan sát được. Mô hình và baseline phải dùng cùng tập hàng trong từng bảng so sánh.

### 8.2. Chỉ số

- Chính: Macro-F1 với danh sách lớp cố định [0, 1, 2].
- Phụ: Accuracy, Precision/Recall/F1/support từng lớp; balanced accuracy nếu cần.
- Confusion matrix số lượng và chuẩn hóa theo hàng (hàng là nhãn thật).
- Theo dõi Recall lớp Cao và tỷ lệ mẫu Cao bị dự đoán Thấp.
- Phân tích theo trạm, mùa/tháng và giờ; các lát cắt ít mẫu phải nêu support.
- Trường hợp nhãn thiếu trong tập đánh giá: xử lý zero_division rõ ràng và báo support, không bỏ lớp để làm đẹp macro-F1.

### 8.3. Bảng thực nghiệm cần điền bằng kết quả thật

| Run | Đặc trưng | C | Weight | Val Macro-F1 | Recall Cao | Fit time | Ghi chú |
|---|---|---|---|---|---|---|---|
| B0 | Majority | — | — | Chưa chạy | Chưa chạy | Chưa chạy | Baseline |
| B1 | Persistence | — | — | Chưa chạy | Chưa chạy | Chưa chạy | Baseline |
| L1 | Cơ bản | 1 | None | Chưa chạy | Chưa chạy | Chưa chạy | Không lag |
| L2 | Cơ bản + lag | 1 | None | Chưa chạy | Chưa chạy | Chưa chạy | MVP |
| L3 | Cơ bản + lag + rolling | 1 | None | Chưa chạy | Chưa chạy | Chưa chạy | Mở rộng |

Chọn theo Macro-F1 validation; nếu gần ngang nhau, ưu tiên cấu hình đơn giản và xem Recall Cao. Không lựa chọn lại bằng test. Nếu không vượt persistence, phân tích những giờ chuyển mức và kiểm tra liệu mô hình chủ yếu tận dụng tính ổn định theo giờ.

### 8.4. Phân tích hệ số

Trích xuất tên đặc trưng sau preprocessing và hệ số theo lớp. Xem các hệ số lớn dương/âm trên biến số đã chuẩn hóa. Không khẳng định hệ số là quan hệ nhân quả; tương quan giữa PM2.5 hiện tại và lag có thể làm hệ số khó diễn giải. Không so tuyệt đối hệ số one-hot và hệ số biến liên tục như cùng một thang đo.

## 9. Cấu trúc dự án đề xuất

| Đường dẫn | Trách nhiệm |
|---|---|
| data/raw/ | CSV gốc, không sửa |
| data/processed/ | Bảng có đặc trưng, target và metadata |
| configs/module2.json | Cấu hình cột, thời gian, tham số |
| notebooks/01_eda.ipynb | Khảo sát dữ liệu |
| notebooks/02_module2_experiments.ipynb | Thực nghiệm và biểu đồ |
| src/module2/data.py | Đọc, kiểm tra, làm sạch dữ liệu |
| src/module2/features.py | Lag, rolling, thời gian; dùng chung train/inference |
| src/module2/labels.py | Fit/apply quy tắc nhãn |
| src/module2/train.py | Pipeline và lựa chọn mô hình |
| src/module2/evaluate.py | Baseline, metric, phân tích lỗi |
| src/module2/predict.py | API phục vụ ứng dụng |
| artifacts/module2/ | model.joblib, metadata.json, label_policy.json |
| reports/module2/ | Bảng metric, hình, dự đoán test |
| tests/ | Kiểm tra rò rỉ, nhãn, hợp đồng dự đoán |
| app.py hoặc pages/ | Giao diện dùng chung của nhóm |
| requirements.txt | Phiên bản thư viện tái lập |

Thư viện dự kiến: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit; pytest cho các kiểm tra quan trọng. Không cần GPU cho kế hoạch này. Chỉ khóa phiên bản sau khi đã chạy thành công môi trường thực tế.

## 10. Hợp đồng tích hợp và lưu mô hình

### 10.1. Thiết kế hàm

```python
def build_features(history_df, station, as_of_time):
    """Chỉ đọc dữ liệu của station có timestamp <= as_of_time."""

def predict_pollution_class(input_data):
    """Nhận một hàng đặc trưng theo schema, trả nhãn/xác suất/metadata."""
```

Hàm thứ nhất cần đủ lịch sử theo cấu hình (t−3..t cho MVP, t−24..t cho bản mở rộng). Thiếu quan sát được pipeline xử lý theo quy tắc huấn luyện; thiếu hẳn cột schema phải báo lỗi. Nêu số giá trị thiếu trong kết quả, không âm thầm nhận nhầm đơn vị.

Ví dụ đầu ra minh họa, không phải dự đoán đã chạy:

```json
{
  "station": "Aotizhongxin",
  "as_of_time": "2016-05-01T09:00:00+08:00",
  "target_time": "2016-05-01T10:00:00+08:00",
  "predicted_class": "Cao",
  "probabilities": {"Thấp": 0.10, "Trung bình": 0.25, "Cao": 0.65},
  "label_policy": "train_quantiles_v1",
  "model_version": "module2_v1",
  "missing_feature_count": 0
}
```

Ánh xạ predict_proba bằng `model.classes_`, không tự giả định thứ tự. Tổng xác suất gần 1. Trạm ngoài tập huấn luyện phải được thông báo chưa được đánh giá; MVP giới hạn lựa chọn trong các trạm đã biết.

### 10.2. Metadata phải lưu

- Danh sách cột, kiểu, đơn vị và thứ tự đầu vào.
- Mốc chia tập, danh sách trạm, thời điểm cuối dữ liệu huấn luyện.
- Ngưỡng và tên lớp; không tính lại khi mở ứng dụng.
- Lag/rolling, phương pháp xử lý thiếu, quy tắc timestamp.
- Hyperparameter, phiên bản thư viện, checksum dữ liệu, phiên bản mã nếu có.
- Metric validation/test và timestamp tạo artifact.

## 11. Giao diện Streamlit

MVP nên dùng dữ liệu lịch sử để tránh bắt người dùng nhập nhiều giá trị lag.

1. Người dùng chọn trạm.
2. Chọn thời điểm t trong giai đoạn test, sau mốc dữ liệu dùng để huấn luyện model đang tải.
3. Hệ thống chỉ lấy lịch sử đến t rồi tạo đặc trưng.
4. Nhấn Dự đoán giờ tiếp theo.
5. Hiển thị target_time, nhóm, biểu đồ xác suất và quy tắc phân loại.
6. Tùy chọn mở phần so sánh số đo/nhãn thật t+1 sau khi dự đoán; dữ liệu này không được đi vào hàm feature.
7. Hiển thị lịch sử PM2.5 gần nhất và thông báo dữ liệu thiếu nếu có.

Gắn mô tả “Phân nhóm nồng độ PM2.5 theo phân vị tập huấn luyện; không phải AQI chính thức” nếu dùng MVP. Không trình bày dataset lịch sử như số liệu không khí thời gian thực.

## 12. Kiểm tra bắt buộc

| Kiểm tra | Điều kiện đạt |
|---|---|
| Lag theo trạm | Lag đầu trạm không lấy giá trị cuối trạm trước |
| Khoảng giờ khuyết | Lag 1 nghĩa là đúng t−1h, không phải dòng liền trước bất kỳ |
| Nhãn tương lai | target_time đúng t+1h; đối chiếu số đo gốc |
| Nhãn thiếu | Không được tạo nhãn từ số đo đã điền thiếu |
| Biên ngưỡng | q1 thuộc Thấp, q2 thuộc Trung bình, lớn hơn q2 thuộc Cao |
| Phân chia | target_time các tập không giao nhau |
| Tính nhân quả | Sửa các dòng sau t không làm thay đổi X_t hay dự đoán tại t |
| Preprocessing | Chỉ fit trên phần dữ liệu được phép của lần chạy |
| X đầu vào | Không chứa target, nhãn hoặc thông tin tương lai |
| Lưu/tải | Dự đoán trước và sau load model trùng trong sai số số học |
| Xác suất | Ánh xạ đúng classes_, không âm và tổng gần 1 |
| Tích hợp | Notebook và app tạo cùng đặc trưng từ cùng lịch sử |

Ưu tiên các kiểm tra này vì chúng xác nhận tính đúng của bài toán, không chỉ kiểm tra code có chạy hay không.

## 13. Lộ trình 8 buổi

Ước lượng 2–4 giờ/buổi, tùy nền tảng và việc thống nhất quy tắc nhãn. Nếu phải triển khai AQI chính thức, dành thêm thời gian xác minh tiêu chuẩn.

| Buổi | Công việc | Đầu ra / điều kiện hoàn thành |
|---|---|---|
| 1 | Chốt phạm vi, tải dữ liệu, audit schema | Đọc được đúng bộ CSV, có báo cáo dữ liệu |
| 2 | Tạo timestamp/lưới giờ, chia tập, chính sách nhãn | Có target đúng, ngưỡng được lưu |
| 3 | Tạo lag, bộ đặc trưng, preprocessing | Qua kiểm tra chống rò rỉ |
| 4 | Baseline và Logistic Regression ban đầu | Có metric validation và confusion matrix |
| 5 | Thử C, weight, bộ đặc trưng | Có bảng thực nghiệm, chọn cấu hình |
| 6 | Khóa cấu hình, refit nếu cần, đánh giá test | Báo cáo cuối và dự đoán từng mẫu |
| 7 | Lưu artifact, API, Streamlit | Demo chọn trạm/thời điểm chạy được |
| 8 | Kiểm tra tái lập, hoàn thiện báo cáo | Chạy lại từ môi trường sạch và bàn giao |

Nếu thiếu thời gian: giữ ba lớp phân vị, bỏ rolling/tuning mở rộng, vẫn giữ baseline, chia theo thời gian và kiểm tra rò rỉ. Không cắt các bước này để ưu tiên giao diện.

## 14. Phối hợp với ba module còn lại

| Đối tác | Nội dung cần thống nhất |
|---|---|
| Phần dữ liệu chung | Timestamp, trạm, đơn vị, làm sạch, mốc chia và horizon |
| Module 1/4 | Cùng PM2.5 mục tiêu tại t+1; có thể dùng chung feature builder |
| Module 3 | Dùng cùng tên trạm/cột; quy tắc truy xuất không làm lộ tương lai trong demo |
| Người tích hợp app | Schema API, model path, lịch sử cần cung cấp và nhãn hiển thị |

Module 2 không phụ thuộc đầu ra module 1. Không so trực tiếp Accuracy của module 2 với RMSE module 1/4. Có thể thêm thí nghiệm phân nhóm nồng độ dự báo của module 1/4 bằng cùng label_policy để so classification, nhưng đây là phần mở rộng trên cùng test, không phải yêu cầu MVP.

## 15. Rủi ro và hướng xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Gọi sai nhãn là AQI | Nêu chính xác chính sách nhãn; chuẩn hóa tiêu chuẩn trước bản nghiệm thu |
| Độ chính xác cao nhờ nhìn đáp án | Kiểm tra danh sách X và thay đổi dữ liệu sau t |
| Kết quả cao chỉ vì không khí ổn định | So với persistence và phân tích giờ chuyển mức |
| Thiếu lớp/mất cân bằng theo giai đoạn | Báo support, Macro-F1, thử class_weight |
| Không hội tụ | Kiểm tra scaler, giá trị lỗi, tăng max_iter; ghi log |
| Train/app tạo feature khác nhau | Dùng chung một hàm feature builder |
| Dữ liệu trạm mới khác phân phối | Giới hạn demo trạm đã học; chưa tuyên bố khái quát sang Việt Nam |
| Nhãn đổi sau khi đã làm | Tăng version, huấn luyện lại và thay đồng bộ metadata/UI |

## 16. Checklist nghiệm thu

- [ ] Nguồn dữ liệu và giả định được ghi rõ.
- [ ] Không nhầm DEWP với độ ẩm, không dùng No làm đặc trưng.
- [ ] Có bảng ngưỡng nhãn, thời gian fit và tên lớp chính xác.
- [ ] Target đúng trạm, đúng t+1; không điền thiếu nhãn.
- [ ] Mốc train/validation/test theo thời gian được lưu.
- [ ] Có majority và persistence baseline trên cùng mẫu.
- [ ] Có Accuracy, Macro-F1, báo cáo từng lớp, confusion matrix.
- [ ] Có phân tích Recall Cao và lỗi Cao → Thấp.
- [ ] Pipeline và metadata tải lại được.
- [ ] Demo chỉ dùng thông tin đến t để dự đoán t+1.
- [ ] Có hướng dẫn tái chạy, phiên bản môi trường, bảng kết quả thật.
- [ ] Báo cáo nêu hạn chế, không bịa kết quả hoặc bảo đảm mô hình vượt baseline.

## 17. Dàn ý báo cáo và câu hỏi bảo vệ

**Dàn ý:** Bài toán → Dữ liệu → Định nghĩa nhãn → Đặc trưng → Chia tập → Logistic Regression → Baseline → Thực nghiệm → Phân tích lỗi → Demo → Hạn chế.

Chuẩn bị trả lời:

1. Vì sao gọi là dự báo thay vì phân loại số đo hiện tại?
2. Nhãn được tạo bằng quy tắc nào, có phải AQI không?
3. Vì sao không chia ngẫu nhiên?
4. Vì sao cần chuẩn hóa cho Logistic Regression?
5. Macro-F1 có lợi gì so với Accuracy trong bài này?
6. Có được dùng PM2.5(t) không? Có, nếu số đo đã sẵn sàng lúc dự báo; PM2.5(t+1) thì không.
7. Mô hình có tốt hơn giữ nguyên mức hiện tại không?
8. Nếu lớp Cao bị dự đoán Thấp, điều đó thể hiện ở đâu trong confusion matrix?
9. Có áp dụng kết quả tại Bắc Kinh sang Hà Nội ngay được không? Chưa, cần dữ liệu và đánh giá miền đích.

## 18. Nguồn và giới hạn của kế hoạch

- **[S0]** `De_tai_du_doan_chat_luong_khong_khi.docx`: tài liệu đề tài người dùng cung cấp, phần Module 2 và quy trình chung.
- **[S1]** UCI — Beijing Multi-Site Air Quality: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data — metadata, schema, thời gian, nguồn và giấy phép; truy cập 21/10/2026.
- **[S2]** scikit-learn — LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html — tham khảo API và tham số; truy cập 21/10/2026. Khi code cần đối chiếu phiên bản cài thực tế.

Các lựa chọn phân vị, ngày chia tập, cấu trúc mã, lịch thực hiện và checklist là thiết kế đề xuất cho dự án này, không phải yêu cầu nguyên văn của UCI. Kế hoạch chưa bao gồm kết quả EDA thực tế, ngưỡng phân vị bằng số hoặc metric mô hình; chúng phải được sinh từ lần triển khai.
