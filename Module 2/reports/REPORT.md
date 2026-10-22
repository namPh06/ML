# Báo cáo Module 2 Logistic Regression

Dự báo nhóm nồng độ PM2.5 tại t+1 giờ, dùng số đo đã có đến t. Đây là dự báo cuốn chiếu; không phải AQI chính thức hoặc khuyến cáo sức khỏe.

Dữ liệu: 420,768 dòng, 12 trạm. Số mẫu train/validation/test: 205,977 / 103,287 / 102,753.
Ngưỡng train: q1=34, q2=96 µg/m³. Thấp: p≤q1; Trung bình: q1<p≤q2; Cao: p>q2.
Cấu hình chọn bằng validation: rolling, C=10.0, class_weight=balanced. Không refit train+validation.

| Mô hình | Accuracy | Macro-F1 | Recall Cao | Cao → Thấp |
|---|---:|---:|---:|---:|
| majority | 0.3795 | 0.1834 | 0.0000 | 1.0000 |
| persistence | 0.8951 | 0.8949 | 0.9069 | 0.0100 |
| logistic | 0.8970 | 0.8969 | 0.8889 | 0.0025 |

Chênh lệch Macro-F1 Logistic so với persistence: +0.0020. Persistence mạnh vì nồng độ thường ổn định giữa hai giờ liền nhau.
Tỷ lệ fallback persistence trên test: 0.89%. So sánh trên cùng tập con có PM2.5(t) nằm trong metrics.csv.
Có 10,221 giờ chuyển lớp khi PM2.5(t) quan sát được; Macro-F1 Logistic tại các giờ này: 0.2534.

## Phương pháp và diễn giải
Logistic Regression đa lớp dùng softmax P(y=k|x)=exp(w_k·x+b_k)/Σ_j exp(w_j·x+b_j). C nhỏ tăng điều chuẩn L2. Numeric được điền median và chuẩn hóa chỉ từ train; station và wd dùng one-hot. Macro-F1 là trung bình F1 của ba lớp, kể cả khi một lát cắt thiếu lớp.
Lag theo từng trạm trên lưới giờ; không nội suy mục tiêu. Chia theo target_time để nhãn train không tràn sang validation. Thử ba bộ đặc trưng ở C=1 rồi tối đa tám cấu hình C/weight trên bộ được chọn; chỉ đánh giá test sau khi khóa cấu hình.
Hệ số dương làm tăng logit của lớp tương ứng khi giữ các biến khác cố định. Hệ số numeric gắn với một độ lệch chuẩn; không so trực tiếp với one-hot và không suy diễn nhân quả. PM2.5 và các lag tương quan nên hệ số có thể khó diễn giải.

## Hạn chế
Nhãn phân vị là mức tương đối tại Bắc Kinh, không phải mức an toàn sức khỏe. Mẫu thiếu mục tiêu bị loại có thể gây thiên lệch. Xác suất chưa được hiệu chỉnh. Chưa đánh giá trạm mới, Việt Nam hoặc dữ liệu thời gian thực. Mô hình tuyến tính có thể bỏ lỡ biến động đột ngột.
Nguồn: Chen, S. (2017), UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G, CC BY 4.0. CSV được cung cấp sẵn; không xác định được thời điểm tải gốc. SHA-256 được lưu trong data_audit.json.

## Tệp kết quả
experiments.csv: toàn bộ lượt huấn luyện và cảnh báo; classification_report.json: precision/recall/F1/support; test_slices.csv: trạm/tháng/giờ và support; test_predictions.csv: nhãn, baseline và xác suất từng mẫu; coefficients.csv: hệ số theo lớp.

## Phân tích bổ sung

Recall Cao của Logistic là 88,89%, thấp hơn persistence 90,69%, dù Macro-F1 cao hơn. Lỗi Cao → Thấp giảm từ khoảng 1,00% xuống 0,25%; cần cân nhắc cả hai khi trình bày kết quả.

Theo các lát cắt test, Gucheng có Macro-F1 thấp nhất trong các trạm (0,8895; 8.605 mẫu). Tháng 6 thấp nhất trong các tháng (0,8524; 8.433 mẫu); giờ 9 thấp nhất trong các giờ (0,8828; 4.302 mẫu). Đây là phân tích sau đánh giá, không dùng để chọn lại mô hình.
