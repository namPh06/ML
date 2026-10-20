# ML — Đề tài chất lượng không khí

Module 3 dùng KNN để tìm các thời điểm lịch sử có điều kiện không khí tương tự, với dữ liệu đủ 12 trạm Bắc Kinh. Sau khi clone/pull, không cần chép mô hình từ máy người viết.

## Chạy lần đầu trên Windows

Cài Python 3.11–3.14. Mở terminal ở thư mục repo (cùng cấp với README này và run_module3.py):

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_module3.py
~~~

Mở địa chỉ Streamlit hiện trong terminal, mặc định http://localhost:8501. Ctrl+C để dừng. Không cần kích hoạt venv nếu dùng đúng đường dẫn Python như trên.

## macOS / Linux

~~~bash
python3 -m venv .venv
.venv/bin/python -X utf8 -m pip install -r requirements.txt
.venv/bin/python run_module3.py
~~~

## Sau mỗi lần pull

Nếu requirements thay đổi, chạy lại lệnh pip install trong môi trường ảo. Sau đó chạy lại run_module3.py. Lệnh khởi chạy kiểm tra phiên bản thư viện và chỉ rõ lệnh cài nếu thiếu/sai phiên bản.

Lần đầu, chương trình tự đọc 12 CSV PRSA ở data/, xử lý dữ liệu, tạo hai bộ tìm kiếm và báo cáo rồi mở giao diện. Khi dữ liệu, mã xử lý hoặc thư viện thay đổi, artifact được tạo lại. Nếu còn khớp, chương trình kiểm tra nạp/truy vấn và dùng lại artifact. Lần đầu cần chờ bước chuẩn bị hoàn tất; chưa mở giao diện ngay lập tức.

Không cần mạng để chuẩn bị dữ liệu nếu đã có 12 CSV trong repo. Cài thư viện cần Internet hoặc pip cache. Nếu không có CSV chung và không có ZIP cache, chương trình tải dataset công khai từ UCI và cần Internet.

## Tùy chọn

Các ví dụ sau dùng python của venv đã kích hoạt; nếu chưa kích hoạt, thay bằng đường dẫn Python của venv như trên.

~~~powershell
python run_module3.py --prepare-only
python run_module3.py --rebuild
python run_module3.py --port 8502
python run_module3.py --data "duong-dan/du-lieu-PRSA"
~~~

Có thể chạy launcher bằng đường dẫn tuyệt đối từ thư mục khác. Giá trị --data tương đối được tính từ terminal đang đứng. Artifact mặc định luôn nằm trong Module 3/artifacts/module3, không phụ thuộc terminal.

## Cấu trúc

~~~text
ML/
├── run_module3.py          # Chuẩn bị rồi mở ứng dụng
├── requirements.txt       # Thư viện runtime được chốt phiên bản
├── data/                  # 12 CSV PRSA chung đã có trong Git
└── Module 3/
    ├── module3/           # Xử lý dữ liệu, KNN, chuẩn bị, app và notebook
    ├── requirements.txt
    ├── requirements-notebook.txt
    ├── data/raw/          # ZIP cache, tự tải khi cần, không đưa vào Git
    └── artifacts/module3/ # Tự sinh trên từng máy, không đưa vào Git
~~~

Hướng dẫn, API ghép nhóm và phương pháp: [Module 3/README_MODULE3.md](Module%203/README_MODULE3.md).

Nguồn và nội dung dữ liệu: [data/README.md](data/README.md). Hai file data.csv và test.csv tại data/ là dữ liệu giá giao dịch khác, không được module 3 sử dụng.

## Notebook

~~~powershell
python -X utf8 -m pip install -r "Module 3/requirements-notebook.txt"
~~~

Mở Module 3/module3/module3_experiments.ipynb bằng VS Code/Jupyter, chọn interpreter của venv rồi Run All. Notebook tự chuẩn bị artifact khi cần và không phụ thuộc output đã lưu trên máy tác giả.

## Khi gặp lỗi

- Thiếu hoặc sai thư viện: dùng đúng Python của venv để chạy pip install -r requirements.txt rồi chạy lại launcher.
- Cổng 8501 bận: thêm --port 8502 hoặc cổng khác.
- Artifact cũ/hỏng hoặc lần trước bị ngắt: chạy lại launcher; có thể thêm --rebuild để buộc tạo lại.
- Thiếu dữ liệu và máy không vào UCI: giữ 12 CSV PRSA trong data/ của repo, hoặc tải ZIP UCI và đặt vào Module 3/data/raw/beijing_uci_501.zip.
- Tách riêng Module 3: chép cả run_module3.py và requirements.txt cùng cấu trúc, hoặc dùng CLI trong README module.

Không commit venv, cache Python, ZIP tải về hoặc artifact sinh ra. Cần commit/push mã nguồn mới và các file cấu hình/hướng dẫn để người khác pull được các thay đổi này. Chỉ nạp joblib do chính nhóm tạo.
