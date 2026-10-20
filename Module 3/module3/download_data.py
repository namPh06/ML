"""Tải ZIP UCI khi không có sẵn dữ liệu chung trong repo."""
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import argparse
from .paths import DEFAULT_RAW

URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

def validate_zip(content):
    with ZipFile(BytesIO(content)) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive bị lỗi checksum.")
        if not any(name.lower().endswith((".zip", ".csv")) for name in archive.namelist()):
            raise ValueError("Archive không có dữ liệu CSV/ZIP.")

def download_dataset(output=None, force=False):
    output = Path(output) if output is not None else DEFAULT_RAW
    if output.exists() and not force:
        validate_zip(output.read_bytes())
        print(f"Đã có ZIP hợp lệ: {output}", flush=True)
        return output
    request = Request(URL, headers={"User-Agent": "Beijing-Air-Quality-Student-Project/3.0"})
    with urlopen(request, timeout=120) as response:
        content = response.read()
    validate_zip(content)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Đã tải {len(content):,} bytes: {output}", flush=True)
    return output

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Mặc định Module 3/data/raw/beijing_uci_501.zip theo vị trí mã nguồn")
    parser.add_argument("--force", action="store_true", help="Tải lại ZIP, kể cả đã có tệp")
    args = parser.parse_args()
    try:
        download_dataset(args.output, args.force)
    except (OSError, URLError, BadZipFile, ValueError) as exc:
        parser.exit(1, f"Không chuẩn bị được dữ liệu: {exc}\nCó thể đặt 12 CSV PRSA vào data/ ở thư mục repo hoặc tải ZIP UCI theo README.\n")

if __name__ == "__main__":
    main()
