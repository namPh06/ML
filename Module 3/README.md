# Module 3 — Cài đặt và khởi chạy

Module tìm các thời điểm lịch sử có điều kiện không khí tương đồng bằng KNN trên dữ liệu 12 trạm Bắc Kinh. Không cần chép joblib từ máy tác giả.

## Windows

Cài Python 3.11–3.14. Từ thư mục ML, chạy:

~~~powershell
cd "Module 3"
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_module3.py
~~~

Nếu đã đứng trong Module 3 thì bỏ lệnh cd đầu tiên. Nếu đang ở module3 bên trong, chạy cd .. trước. Không cần kích hoạt venv khi dùng đường dẫn Python như trên.

## macOS / Linux

~~~bash
cd "Module 3"
python3 -m venv .venv
.venv/bin/python -X utf8 -m pip install -r requirements.txt
.venv/bin/python run_module3.py
~~~

Mở địa chỉ Streamlit in trong terminal, mặc định http://localhost:8501. Ctrl+C để dừng. Không chạy ứng dụng bằng python module3/app.py vì giao diện cần môi trường Streamlit.

## Cấu trúc

~~~text
ML/
├── data/                       # Dữ liệu dùng chung của nhóm
└── Module 3/
    ├── run_module3.py          # Chuẩn bị rồi mở giao diện
    ├── README.md              # Hướng dẫn cài/chạy
    ├── README_MODULE3.md       # Phương pháp và API tích hợp
    ├── REVIEW_MODULE3.md       # Kết quả rà soát
    ├── requirements.txt        # Thư viện runtime chốt phiên bản
    ├── requirements-notebook.txt
    ├── requirements-tested.txt
    ├── .gitignore
    ├── module3/               # Mã xử lý, KNN, giao diện, notebook
    ├── data/raw/              # ZIP cache tự tải khi cần
    └── artifacts/module3/     # Mô hình và báo cáo tự sinh
~~~

.git được giữ tại ML để quản lý phiên bản. Quy tắc bỏ qua venv, cache và artifact nằm trong Module 3/.gitignore.

## Chuẩn bị và dùng lại mô hình

Lần đầu launcher đọc 12 CSV PRSA trong ../data/, xử lý dữ liệu, tạo hai mô hình và báo cáo rồi mở giao diện. Không cần mạng để chuẩn bị nếu đã có các CSV; cài thư viện cần Internet hoặc pip cache. Nếu thiếu CSV và ZIP cache, chương trình tải ZIP từ UCI.

Các lần sau launcher kiểm tra nội dung dữ liệu, mã pipeline, phiên bản thư viện và khả năng nạp/truy vấn mô hình. Artifact phù hợp được dùng lại; artifact cũ, hỏng hoặc bị gián đoạn khi tạo được xây lại. Bước chuẩn bị có khóa hệ điều hành tránh ghi đồng thời.

Sau khi pull, chạy lại pip install nếu requirements thay đổi, rồi chạy launcher bằng Python của venv.

Các ví dụ sau chạy trong Module 3 bằng Python của venv đã kích hoạt; nếu chưa kích hoạt, dùng đường dẫn Python như trên:

~~~powershell
python run_module3.py --prepare-only
python run_module3.py --rebuild
python run_module3.py --port 8502
python run_module3.py --data "../data"
~~~

Có thể gọi launcher bằng đường dẫn tuyệt đối từ thư mục khác. --data tương đối tính từ terminal đang đứng; artifact mặc định luôn nằm trong Module 3/artifacts/module3.

## Notebook và tích hợp

Từ Module 3, cài bổ sung:

~~~powershell
python -X utf8 -m pip install -r requirements-notebook.txt
~~~

Mở module3/module3_experiments.ipynb, chọn interpreter venv rồi Run All. Notebook tự chuẩn bị mô hình khi cần. Hướng dẫn phương pháp và API: [README_MODULE3.md](README_MODULE3.md). Nguồn dữ liệu: [../data/README.md](../data/README.md). Hai CSV giá giao dịch data.csv và test.csv trong data chung không được module sử dụng.

## Khi gặp lỗi

- Thiếu/sai thư viện: dùng đúng Python của venv để cài requirements.txt rồi chạy lại.
- Cổng 8501 bận: thêm --port 8502.
- Artifact cũ/hỏng: chạy lại launcher; dùng --rebuild để buộc tạo lại.
- Không tải được UCI: giữ 12 CSV PRSA trong ../data/ hoặc đặt ZIP vào data/raw/beijing_uci_501.zip của Module 3.
- Tách riêng Module 3: chép nguyên thư mục này; khi không có ../data/, chương trình dùng ZIP cache hoặc tải từ UCI.

Chỉ nạp joblib do nhóm tự tạo. Không commit venv, cache, ZIP tải về hoặc artifact sinh ra. Cần commit/push thay đổi mã và cấu trúc để người khác pull được.
