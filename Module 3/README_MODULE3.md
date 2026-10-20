# Module 3 — Tìm thời điểm có điều kiện không khí tương đồng

Module độc lập của đề tài: StandardScaler + NearestNeighbors, đủ 12 trạm, khoảng cách Euclidean/Manhattan, tìm cùng trạm hoặc toàn kho. Không cần nhãn y và không dự báo PM2.5 giờ tiếp theo.

## Cài và chạy sau khi clone/pull

Cách khuyến nghị: vào thư mục Module 3, làm theo [hướng dẫn khởi chạy](README.md). Trên Windows:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_module3.py
~~~

Các lệnh trên chạy trong thư mục Module 3, nơi chứa run_module3.py và requirements.txt. Không cần chép joblib hay thư mục artifacts từ máy khác.

Lần đầu launcher tự chuẩn bị dữ liệu/mô hình/báo cáo. Các lần sau kiểm tra chữ ký nội dung dữ liệu, mã pipeline và phiên bản thư viện; chỉ dùng lại artifact phù hợp và nạp/truy vấn được. Bước chuẩn bị có khóa OS để hai launcher không cùng ghi mô hình. Nếu bị ngắt khi chuẩn bị, lần sau tạo lại. Có thể buộc tạo lại bằng --rebuild; --prepare-only chỉ chuẩn bị, chưa mở giao diện.

Python yêu cầu từ 3.11, khuyến nghị 3.11–3.14. requirements.txt chốt các thư viện runtime giống môi trường đã kiểm tra; requirements-notebook.txt bổ sung công cụ notebook.

## Cấu trúc và đường dẫn

~~~text
repo/data/PRSA_Data_*.csv             # Dữ liệu chung, ưu tiên đọc
repo/Module 3/
    run_module3.py                   # Chuẩn bị rồi mở giao diện
    README.md                        # Cài và chạy
    requirements.txt                 # Thư viện runtime
    module3/paths.py                  # Đường dẫn mặc định theo vị trí mã
    module3/prepare.py                # Kiểm tra cache, tải dữ liệu nếu thiếu, xây artifact
    module3/download_data.py          # Tải/kiểm tra ZIP UCI
    module3/data.py                   # Đọc, kiểm tra, lọc thiếu, chia thời gian
    module3/similarity.py             # Scaler, KNN, API truy vấn và save/load
    module3/train.py                  # Xây cả hai mô hình, thực nghiệm và CSV
    module3/reporting.py              # Biểu đồ và báo cáo Markdown
    module3/app.py                    # Giao diện; render() để tích hợp
    module3/module3_experiments.ipynb # Notebook dùng cùng pipeline
    data/raw/beijing_uci_501.zip      # ZIP cache khi cần
    artifacts/module3/               # Tất cả sản phẩm sinh tự động
~~~

Nguồn dữ liệu mặc định ưu tiên data/ của repo nếu có CSV PRSA. Nếu không có, dùng ZIP cache trong module; launcher/notebook tải ZIP từ UCI nếu cần. Bộ đọc thư mục chỉ lấy PRSA_Data_*.csv nên không đọc nhầm hai CSV giá giao dịch trong data/ chung.

CLI mặc định đặt artifact theo vị trí mã nguồn. --data/--output được truyền tường minh dùng đường dẫn tương đối từ thư mục gọi lệnh. Các lệnh dạng python -m module3... cần chạy từ Module 3 hoặc có package module3 trong sys.path. Launcher trong Module 3 tự xử lý việc này.

## Sử dụng giao diện

1. Chọn Euclidean/Manhattan, phạm vi một trạm/toàn kho và trạm truy vấn.
2. Chọn K từ 1 đến 50.
3. Chọn ngày/giờ có sẵn hoặc nhập sáu điều kiện, rồi bấm tìm.
4. Xem bảng thời điểm, khoảng cách, chênh lệch từng biến, biểu đồ và tải CSV.

Kết quả được giữ qua rerun/tải CSV. Đổi đầu vào sẽ báo cần tìm lại. Các tab Dữ liệu, Đánh giá, Cách hiểu kết quả cung cấp nguồn, số lượng và kết quả thực nghiệm.

Chú giải ngay trên bảng ghi tên đầy đủ, đơn vị và cách hiểu sáu biến. Tiêu đề cột có trợ giúp. Mức thấp/cao theo Q25/Q75 của train chỉ là thống kê trong kho. Bảng WHO ghi rõ thời gian lấy trung bình, ví dụ và liên kết nguồn; không dùng ngưỡng ngày/năm để gán nhãn sức khỏe cho từng bản ghi theo giờ. Bảng Δ giải thích dấu và giữ nguyên đơn vị gốc.

## Dữ liệu và phương pháp

Nguồn: [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality), DOI 10.24432/C5RK5G, CC BY 4.0. Bộ gốc có 420.768 dòng của 12 trạm, 01/03/2013–28/02/2017. Thời gian là giờ Bắc Kinh UTC+8, lưu datetime không gắn timezone.

Thứ tự đặc trưng: PM2.5, PM10, TEMP, PRES, DEWP, WSPM. PM2.5/PM10: µg/m³; TEMP/DEWP: °C; PRES: hPa; WSPM: m/s. DEWP là điểm sương, không phải độ ẩm tương đối. station/datetime chỉ là định danh, không tính vào khoảng cách.

- Bỏ định danh thiếu/sai. Chấp nhận datetime hoặc year/month/day/hour.
- Chuyển đặc trưng thành số. Loại thiếu, không phải số, vô hạn, nồng độ/gió âm, áp suất không dương. Giữ nhiệt độ/điểm sương âm hợp lệ.
- Bỏ dòng trùng khóa và đặc trưng; báo lỗi khi trùng station/datetime nhưng khác đặc trưng.
- Bỏ hàng thiếu một trong sáu biến; không nội suy hoặc điền bằng dữ liệu tương lai.
- Train trước 01/11/2015; validation trước 01/07/2016; test từ 01/07/2016. Mỗi trạm phải có dữ liệu trong cả ba tập.
- Chỉ fit StandardScaler trên train. Dùng một scaler chung và chỉ mục KNN riêng cho toàn kho/từng trạm; không fit lại theo truy vấn.

Validation khảo sát Euclidean/Manhattan × K=3/5/10 × 2 phạm vi = 12 cấu hình. Test chỉ báo cáo cấu hình đặt trước Euclidean, K=5 với hai phạm vi. Mặc định 120 truy vấn/lần, cân bằng 10/trạm, seed=42. Đối chứng ngẫu nhiên lấy cùng K và cùng phạm vi. Không chọn metric chỉ bằng cách so trực tiếp trị số khoảng cách giữa các metric.

Dữ liệu hiện hành: 411.411 dòng đủ đặc trưng; 9.357 dòng bị loại; train 274.601, validation 68.498, test 68.312. Kết quả chi tiết được tạo trên từng máy; thời gian truy vấn tùy máy.

## Artifact tự sinh

Tất cả nằm ở artifacts/module3/ của module, được gitignore:

| Tệp | Nội dung |
|---|---|
| knn.joblib, knn_manhattan.joblib | Hai bộ tìm kiếm; launcher mặc định Euclidean là bộ chính |
| metadata.json | Trạm, đặc trưng, mốc chia, phiên bản, checksum và chữ ký chuẩn bị |
| validation.csv, test.csv | Các thời điểm sau train, không tham gia kho tìm kiếm |
| data_quality_by_station.csv, split_summary.csv | Chất lượng dữ liệu và chia tập theo từng trạm |
| validation_metrics.csv | 12 cấu hình khảo sát |
| test_metrics.csv, test_query_details.csv | Tổng hợp và chi tiết truy vấn test |
| test_metrics_by_station.csv | Đánh giá test theo 12 trạm × 2 phạm vi |
| example_queries.csv, example_neighbors.csv | Ví dụ truy vấn/hàng xóm |
| station_coverage.png, validation_comparison.png, example_comparison.png | Biểu đồ dữ liệu, thực nghiệm, ví dụ |
| REPORT_MODULE3.md | Báo cáo tự động từ CSV thực nghiệm |

Không cần đưa artifact hoặc ZIP cache vào Git. Commit/push mã nguồn, cấu hình, hướng dẫn và các CSV PRSA chung. Máy mới tự sinh mô hình bằng cùng pipeline.

## API ghép nhóm

Dùng đơn vị gốc; không đưa bảng đã qua scaler của module khác vào. Nạp một lần và tái sử dụng:

~~~python
from module3.paths import DEFAULT_ARTIFACTS
from module3.similarity import SimilaritySearch, find_similar_periods

searcher = SimilaritySearch.load(DEFAULT_ARTIFACTS / "knn.joblib")
query = {"PM2.5": 100, "PM10": 140, "TEMP": 25,
         "PRES": 1010, "DEWP": 18, "WSPM": 1, "station": "Aotizhongxin"}
results = find_similar_periods(query, k=5, searcher=searcher)
results_all = find_similar_periods(query, k=5, searcher=searcher, station="ALL")
results_wanliu = searcher.find_similar_periods(query, k=5, station="Wanliu")
~~~

Đầu ra DataFrame có rank, datetime, station, sáu biến theo đơn vị gốc và distance tăng dần. K nguyên dương, không vượt số ứng viên; API không giới hạn 50 như giao diện. Khi bằng khoảng cách có thể có nhiều hàng xóm hợp lệ.

station=None theo trạm của query; query không có station thì tìm toàn kho. station="ALL" luôn tìm toàn kho. Nhập tay có thể bỏ datetime để so với kho cố định. Nếu có datetime và past_only=True, truy vấn phải sau toàn bộ kho train để scaler không học từ tương lai. past_only=False dành cho duyệt kho, cần station để loại đúng bản ghi truy vấn; không dùng để mô phỏng quá khứ.

Từ ứng dụng nhóm ở repo, thêm thư mục Module 3 vào sys.path theo vị trí file ứng dụng chính, sau đó import render:

~~~python
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Module 3"))
from module3.app import render
# Ứng dụng chính gọi st.set_page_config trước; chạy launcher --prepare-only trước khi ghép.
render()
~~~

Các thành viên cần thống nhất quy tắc thiếu và mốc chia trước khi chốt kết quả chung.

## CLI nâng cao và notebook

Từ thư mục Module 3, sau khi cài thư viện:

~~~powershell
python -m module3.prepare
python -m module3.prepare --rebuild
python -m module3.train --data "../data" --output artifacts/experiment
python -m module3.train --station Aotizhongxin --output artifacts/one_station
python -m module3.download_data --force
~~~

train.py cho phép --metric manhattan; lúc đó knn.joblib là Manhattan, file phụ là knn_euclidean.joblib. App đọc model_files trong metadata. Launcher quản lý cache mặc định Euclidean; muốn giữ thí nghiệm riêng nên dùng --output riêng với train.py.

Cài bổ sung notebook từ Module 3: python -X utf8 -m pip install -r requirements-notebook.txt. Chọn interpreter venv, mở notebook và Run All. REBUILD=True buộc tạo lại; mặc định dùng lại artifact phù hợp. Output notebook được xóa khỏi bản nguồn để không mang theo ảnh/kết quả cũ của máy tác giả; chạy sẽ tạo kết quả thực.

## Giới hạn

Tìm quan sát theo giờ, chưa tìm cả chuỗi nhiều giờ; các giờ liền nhau có thể cùng xuất hiện. Chưa học trọng số quan trọng của từng biến. PM2.5 là đầu vào, nên hàng xóm gần về PM2.5 không phải đánh giá độc lập. Không dùng Accuracy, AQI hoặc phần trăm tương đồng. KNN trả đủ K nếu đủ mẫu dù chúng có thể xa kho. Dữ liệu thiếu bị loại có thể làm lệch mẫu; tìm xuyên trạm chưa xét địa lý/nguồn phát thải. Dữ liệu Bắc Kinh không mặc nhiên đại diện Việt Nam.

Chỉ nạp joblib do nhóm tự tạo. Nếu clone xong mở trực tiếp app.py thay vì launcher, cần chạy bước chuẩn bị trước; app sẽ chỉ hướng dẫn khi thiếu artifact.
