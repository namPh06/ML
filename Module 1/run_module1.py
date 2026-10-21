"""Chạy notebook huấn luyện, lưu kết quả, rồi mở ứng dụng Streamlit."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

MODULE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = MODULE_DIR / "module1.ipynb"


def run_notebook():
    """Thực sự chạy từng ô code và lưu kết quả vào chính module1.ipynb."""
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    # Xóa kết quả cũ để các ô chưa chạy không hiển thị kết quả của lần trước.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    client = NotebookClient(
        notebook, kernel_name="python3", timeout=300,
        resources={"metadata": {"path": str(MODULE_DIR)}},
    )
    # Kernel dùng đúng Python hiện tại; nbclient tự đóng kernel khi chạy xong.
    client.create_kernel_manager().kernel_spec.argv[0] = sys.executable
    try:
        # Hồ sơ Jupyter chỉ dùng tạm, không cần ghi vào thư mục cá nhân.
        with TemporaryDirectory(prefix="module1-jupyter-") as profile:
            client.execute(env={**os.environ, "IPYTHONDIR": profile})
    finally:
        # Nếu có lỗi, lưu cả ô báo lỗi để mở notebook xem được nguyên nhân.
        nbformat.write(notebook, NOTEBOOK_PATH)

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text, end="", flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true", help="Chạy notebook và lưu kết quả, không mở ứng dụng")
    parser.add_argument("--port", type=int, default=8502, help="Cổng Streamlit, mặc định 8502")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Cần Python 3.11 trở lên.")
    if not 1 <= args.port <= 65535:
        parser.error("Cổng phải nằm trong 1..65535.")
    try:
        print("[1/3] Đang chạy module1.ipynb để huấn luyện và đánh giá...", flush=True)
        run_notebook()
        print("[2/3] Đã lưu kết quả vào notebook và thư mục outputs.", flush=True)
        if args.train_only:
            return 0
        print(f"[3/3] Mở ứng dụng tại http://127.0.0.1:{args.port}", flush=True)
        return subprocess.call([
            sys.executable, "-B", "-X", "utf8", "-m", "streamlit", "run",
            str(MODULE_DIR / "app.py"), "--server.address=127.0.0.1",
            f"--server.port={args.port}", "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ], cwd=MODULE_DIR)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Không chạy được Module 1: {exc}", file=sys.stderr)
        print("Nếu thiếu thư viện, cài requirements.txt bằng đúng Python đang chạy.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
