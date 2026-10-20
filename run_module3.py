"""Chạy từ thư mục repo hoặc bằng đường dẫn tuyệt đối: python run_module3.py."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODULE_ROOT = ROOT / "Module 3"

def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Chuẩn bị dữ liệu/mô hình, chưa mở Streamlit")
    parser.add_argument("--rebuild", action="store_true", help="Buộc tạo lại mô hình và báo cáo")
    parser.add_argument("--data", help="Nguồn dữ liệu thay thế: CSV, ZIP hoặc thư mục PRSA")
    parser.add_argument("--queries", type=int, default=120, help="Số truy vấn lấy mẫu cho mỗi đánh giá khi chuẩn bị")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.exit(1, "Cần Python 3.11 trở lên (khuyến nghị 3.11–3.14).\n")
    if args.queries < 1 or not 1 <= args.port <= 65535:
        parser.error("--queries phải dương; --port phải trong 1..65535")
    missing = []
    for line in (MODULE_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.strip().split("==")
        try:
            if importlib.metadata.version(name) != expected:
                missing.append(f"{name}=={expected}")
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{name}=={expected}")
    if missing:
        print("Cần cài/cập nhật thư viện: " + ", ".join(missing), file=sys.stderr)
        print(f'Chạy: "{sys.executable}" -X utf8 -m pip install -r "{ROOT / "requirements.txt"}"', file=sys.stderr)
        return 1
    command = [sys.executable, "-B", "-X", "utf8", "-m", "module3.prepare", "--queries", str(args.queries)]
    if args.data:
        command += ["--data", str(Path(args.data).resolve())]
    if args.rebuild:
        command.append("--rebuild")
    try:
        subprocess.run(command, cwd=MODULE_ROOT, check=True)
        if args.prepare_only:
            print("Đã sẵn sàng. Chạy python run_module3.py để mở giao diện.")
            return 0
        return subprocess.call([sys.executable, "-B", "-m", "streamlit", "run",
                                str(MODULE_ROOT / "module3" / "app.py"),
                                "--server.address", args.host, "--server.port", str(args.port),
                                "--browser.gatherUsageStats", "false"], cwd=MODULE_ROOT)
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as exc:
        print("Chuẩn bị chưa thành công; xem thông báo phía trên, sửa nguyên nhân rồi chạy lại.", file=sys.stderr)
        return exc.returncode
    except OSError as exc:
        print(f"Không chạy được module 3: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
