# Rà soát module 3

Ngày rà soát: 20/10/2026.

## Kết luận

Module 3 đáp ứng phần chức năng độc lập trong đề tài: notebook khảo sát khoảng cách và K, API find_similar_periods, bảng thời điểm tương đồng và biểu đồ so sánh. Trong các trường hợp đã kiểm tra, chưa phát hiện lỗi chức năng cần sửa. Có thể dùng để chạy demo và bàn giao cho nhóm tích hợp. Đây không phải xác nhận ứng dụng chung của cả bốn module đã hoàn thành.

## Kết quả kiểm tra

12 nhóm kiểm tra đạt; thêm 240 truy vấn kiểm tra tính nhất quán theo K. Không tạo file mã test để lưu lại. Các fixture và artifact tái tạo dùng TemporaryDirectory và được dọn sau khi kiểm tra.

| Nhóm | Nội dung đã xác nhận | Kết quả |
|---|---|---|
| 1 | ZIP gốc hợp lệ, SHA256 khớp metadata, 420.768 dòng và 12 trạm | Đạt |
| 2 | Chia thời gian không chồng lấn; scaler mean/variance và toàn bộ hàng trong kho khớp train | Đạt |
| 3 | Euclidean/Manhattan được đối chiếu bằng phép tính khoảng cách độc lập trên truy vấn từ cả 12 trạm, cùng trạm/toàn kho/trạm chỉ định | Đạt |
| 4 | Từ chối K sai, thiếu biến, NaN/vô hạn, nồng độ âm, áp suất không dương, trạm không hợp lệ, tìm trước khi fit | Đạt |
| 5 | Chặn truy vấn có datetime không nằm sau kho train; chặn datetime sai/mang timezone; loại chính xác bản ghi khi duyệt kho | Đạt |
| 6 | Loại thiếu/sai miền, bỏ dòng trùng hoàn toàn, báo lỗi trùng định danh khác đặc trưng; kiểm tra mốc chia sai | Đạt |
| 7 | Đọc CSV và ZIP lồng cho cùng dữ liệu; save/load và API tích hợp trả cùng kết quả | Đạt |
| 8 | Mẫu 120 truy vấn cân bằng 10/trạm; đối chứng ngẫu nhiên cùng phạm vi; chỉ số hữu hạn | Đạt |
| 9 | CSV, metadata, báo cáo, số dòng theo tập/trạm và ảnh PNG nhất quán | Đạt |
| 10 | Streamlit AppTest: 12 trạm, hai chế độ nhập, đổi metric/phạm vi/K, giữ kết quả qua rerun, báo đầu vào thay đổi, chú giải và cảnh báo đúng vị trí | Đạt |
| 11 | Notebook đúng schema; thực thi đủ 9 ô mã trong phiên Python mới không lỗi; các file Python đúng cú pháp | Đạt |
| 12 | Tạo lại cả hai mô hình và đầy đủ báo cáo/biểu đồ từ ZIP gốc trong thư mục tạm, dùng 12 truy vấn mỗi phép đánh giá; kết quả truy vấn khớp mô hình hiện hành | Đạt |

Kiểm tra bổ sung: K=1/3/5/10/50 × 12 trạm × 2 metric × 2 phạm vi = 240 truy vấn. Mỗi kết quả đủ K, khoảng cách tăng dần và các khoảng cách đầu danh sách nhất quán khi tăng K. Khi bằng khoảng cách có thể có nhiều hàng xóm hợp lệ.

CLI đã chạy: python -m module3.download_data xác nhận ZIP có sẵn hợp lệ; python -m module3.train --help hiển thị đủ tùy chọn. Không tải lại dữ liệu qua mạng trong lượt rà soát này. Giao diện được kiểm tra bằng Streamlit AppTest; lượt này không kiểm tra bố cục bằng ảnh chụp trình duyệt.

## Dữ liệu hiện hành

| Tập | Số quan sát |
|---|---:|
| Dữ liệu gốc | 420.768 |
| Đủ sáu đặc trưng | 411.411 |
| Loại do thiếu/không hợp lệ | 9.357 |
| Train / kho tìm kiếm | 274.601 |
| Validation | 68.498 |
| Test | 68.312 |

Artifact hiện hành vẫn giữ đánh giá gốc 120 truy vấn mỗi cấu hình. Lượt tái tạo tạm sử dụng 12 truy vấn mỗi cấu hình để kiểm tra pipeline; không ghi đè artifact bàn giao.

## Những điểm cần thống nhất khi ghép nhóm

- Đầu vào dùng đơn vị gốc, gồm station, datetime và PM2.5, PM10, TEMP, PRES, DEWP, WSPM; không đưa dữ liệu đã chuẩn hóa bởi module khác vào API này.
- Dữ liệu lưu giờ Bắc Kinh UTC+8, datetime không gắn timezone.
- Hiện train trước 01/11/2015, validation trước 01/07/2016; quy tắc thiếu là loại hàng thiếu/sai một trong sáu biến. Cần thống nhất với các thành viên trước khi chốt kết quả nhóm.
- Dùng SimilaritySearch.load một lần để gọi API; hoặc đưa module3.app.render() vào trang Streamlit chung. Hướng dẫn và ví dụ ở README_MODULE3.md.
- data/raw và artifacts bị gitignore. Nếu chuyển máy, gửi kèm chúng hoặc chạy lại download_data và train theo README.

## Giới hạn đã ghi rõ trong sản phẩm

Module tìm quan sát theo giờ, chưa tìm chuỗi nhiều giờ; không dự báo giờ tiếp theo, không gán AQI hoặc phần trăm tương đồng. PM2.5 là một đầu vào để tìm gần nhau. KNN trả đủ K dù truy vấn có thể xa kho. Khoảng cách nhỏ không bảo đảm cùng nguồn ô nhiễm. Mức thấp/cao theo Q25/Q75 chỉ là thống kê lịch sử. Mốc WHO hiển thị riêng với thời gian lấy trung bình; không gán nhãn sức khỏe cho bản ghi theo giờ. Tính phù hợp môi trường của hàng xóm vẫn cần người hiểu lĩnh vực đánh giá.

Kết quả kiểm tra phản ánh dữ liệu, thư viện và các trường hợp đã chạy trên máy hiện tại; chưa kiểm tra giao diện ghép của cả nhóm hoặc môi trường triển khai khác.


## Bổ sung bàn giao qua Git — 20/10/2026

Mã đã chuẩn hóa để chạy ở vị trí bất kỳ, không dùng đường dẫn riêng của tác giả. Launcher trong Module 3 kiểm tra thư viện, tự chuẩn bị và mở Streamlit. Mặc định dùng 12 CSV PRSA chung đã có trong Git; không cần chép joblib. Thư viện runtime được chốt phiên bản; công cụ notebook cài riêng.

Đã thử trên bản xuất từ danh sách Git (gồm mã mới chưa commit, loại toàn bộ artifact, ZIP cache và venv) trong đường dẫn có dấu/khoảng trắng. Dùng thư viện đang cài trên máy kiểm tra, không thử tải/cài lại các wheel từ Internet. Các trường hợp đạt:

- Chạy launcher từ ngoài repo, tự tạo đủ artifact từ CSV khi không có mạng.
- Lần chạy kế tiếp dùng lại mô hình, không ghi lại joblib.
- Phát hiện dữ liệu/phiên bản thay đổi và dấu chuẩn bị bị gián đoạn.
- Mô hình hỏng được phát hiện và tự tạo lại từ CSV.
- Giao diện có 12 trạm, hai chế độ nhập, đổi metric/phạm vi, K=50 và chú giải.
- Notebook chạy đủ 9 ô mã, dùng lại artifact vừa tạo.
- Thiếu/sai thư viện được báo cùng lệnh cài, không mở app lỗi.
- Pip dry-run đọc đúng requirements lồng trong đường dẫn có khoảng trắng; cú pháp mã kiểm tra theo Python 3.11.

Bản sao và artifact kiểm tra đã được dọn. Không giữ lại file mã test. README khởi chạy và README module có hướng dẫn cài/chạy Windows và macOS/Linux; chưa trực tiếp chạy trên macOS/Linux. Các thay đổi cần được commit/push trước khi người khác pull được.

## Đóng gói thư mục — 20/10/2026

Toàn bộ launcher, requirements, hướng dẫn, quy tắc ignore, mã nguồn và artifact của Module 3 nằm trong Module 3/. Bên ngoài chỉ còn data/ dùng chung và Module 3/; giữ nguyên .git để quản lý phiên bản. Chạy python run_module3.py từ Module 3. Dữ liệu chung vẫn được tìm theo vị trí mã nguồn.

Kiểm tra sau đóng gói: tạo lại đầy đủ hai mô hình từ CSV chung; gọi launcher bằng đường dẫn từ ngoài Module 3 dùng lại cache; AppTest mở đủ 12 trạm và truy vấn thành công. Đã kiểm tra cú pháp Python 3.11, đường dẫn khởi chạy/cài thư viện và quy tắc gitignore. Không tạo file mã test.
