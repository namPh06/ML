# Dữ liệu dùng chung

12 file PRSA_Data_*.csv là dữ liệu quan trắc theo giờ tại 12 trạm Bắc Kinh, giai đoạn 2013–2017. Mỗi file ứng với một trạm. Nguồn: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality), DOI 10.24432/C5RK5G, CC BY 4.0.

Module 3 chỉ tìm các file theo mẫu PRSA_Data_*.csv. Nó đọc các biến PM2.5, PM10, TEMP, PRES, DEWP, WSPM và định danh thời gian/trạm. Không cần đổi tên hay gộp thủ công các CSV này. Các timestamp dùng giờ Bắc Kinh UTC+8.

Hai file data.csv và test.csv có các cột Date, Open, High, Low, Close, Adj Close, Volume là dữ liệu giá giao dịch khác; module 3 bỏ qua chúng. test.csv ở đây khác với tập test không khí do module 3 sinh tại Module 3/artifacts/module3/test.csv.

Giữ 12 CSV PRSA trong Git để máy mới có thể chuẩn bị mô hình mà không phải tải lại dataset. Không chỉnh giá trị của CSV gốc để sửa lỗi ứng dụng. Mã xử lý thiếu/chia tập nằm trong Module 3/module3/data.py.
