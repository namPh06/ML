"""Đường dẫn mặc định theo vị trí mã nguồn, không theo terminal đang đứng."""
from pathlib import Path
import hashlib

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
DEFAULT_ARTIFACTS = MODULE_ROOT / "artifacts" / "module3"
DEFAULT_RAW = MODULE_ROOT / "data" / "raw" / "beijing_uci_501.zip"

def default_data_source():
    shared = REPO_ROOT / "data"
    if shared.is_dir() and any(shared.glob("PRSA_Data_*.csv")):
        return shared
    return DEFAULT_RAW

def source_fingerprint(source):
    """SHA256 nội dung; thư mục chỉ xét các CSV PRSA giống bộ đọc."""
    source = Path(source)
    if source.is_file():
        return hashlib.sha256(source.read_bytes()).hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(f"Không có dữ liệu: {source}. Chạy python run_module3.py từ thư mục repo.")
    files = sorted(source.rglob("PRSA_Data_*.csv"))
    if not files:
        raise ValueError(f"Không tìm thấy CSV PRSA trong {source}.")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()
