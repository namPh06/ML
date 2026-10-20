"""Chuẩn bị lần đầu và tái sử dụng artifact khi dữ liệu/mã/thư viện còn khớp."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from .paths import MODULE_ROOT, DEFAULT_ARTIFACTS, DEFAULT_RAW, default_data_source, source_fingerprint

REQUIRED_FILES = [
    "metadata.json", "knn.joblib", "knn_manhattan.joblib", "validation.csv", "test.csv",
    "validation_metrics.csv", "test_metrics.csv", "test_query_details.csv",
    "test_metrics_by_station.csv", "split_summary.csv", "data_quality_by_station.csv",
    "example_queries.csv", "example_neighbors.csv", "station_coverage.png",
    "validation_comparison.png", "example_comparison.png", "REPORT_MODULE3.md",
]

def get_data_source(data=None):
    source = Path(data).resolve() if data is not None else default_data_source()
    if data is None and source == DEFAULT_RAW and not source.exists():
        from .download_data import download_dataset
        download_dataset(source)
    if not source.exists():
        raise FileNotFoundError(f"Không có dữ liệu: {source}")
    return source

def preparation_signature(source, queries):
    digest = hashlib.sha256()
    for name in ["data.py", "similarity.py", "train.py", "reporting.py", "paths.py", "prepare.py"]:
        digest.update(name.encode())
        digest.update((MODULE_ROOT / "module3" / name).read_bytes())
    return {
        "schema": 1, "source_sha256": source_fingerprint(source),
        "pipeline_sha256": digest.hexdigest(), "queries": queries,
        "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name) for name in
                      ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib"]},
    }

def artifacts_ready(output, signature):
    output = Path(output)
    if (output / ".preparing").exists() or any(not (output / n).is_file() for n in REQUIRED_FILES):
        return False
    try:
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        return isinstance(metadata, dict) and metadata.get("artifact_version") == 2 and metadata.get("preparation") == signature
    except (ValueError, OSError):
        return False

@contextmanager
def preparation_lock(output, timeout=180):
    """Khóa OS tự nhả khi tiến trình dừng, kể cả bị ngắt giữa chừng."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".prepare.lock")
    with lock.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0"); handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Một tiến trình khác đang chuẩn bị mô hình. Chờ tiến trình đó xong rồi chạy lại.")
                time.sleep(.2)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def prepare(data=None, output=None, force=False, queries=120):
    if isinstance(queries, bool) or not isinstance(queries, int) or queries < 1:
        raise ValueError("queries phải là số nguyên dương.")
    output = Path(output).resolve() if output is not None else DEFAULT_ARTIFACTS
    source = get_data_source(data)
    with preparation_lock(output):
        signature = preparation_signature(source, queries)
        if not force and artifacts_ready(output, signature):
            # Kiểm tra khả năng nạp và truy vấn trước khi mở giao diện.
            from .similarity import SimilaritySearch
            import pandas as pd
            try:
                query = pd.read_csv(output / "test.csv", nrows=1).iloc[0]
                for name in ["knn.joblib", "knn_manhattan.joblib"]:
                    SimilaritySearch.load(output / name).find_similar_periods(query, k=1)
                print("Artifact phù hợp; dùng lại bộ tìm kiếm đã có.", flush=True)
                return json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Không dùng lại được artifact ({type(exc).__name__}); tạo lại từ dữ liệu nguồn.", flush=True)
        print(f"Chuẩn bị module 3 từ: {source}", flush=True)
        print("Lần đầu cần xử lý dữ liệu và xây hai bộ tìm kiếm; những lần sau sẽ dùng lại.", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / ".preparing"
        marker.write_text("incomplete until preparation succeeds", encoding="utf-8")
        from .train import build_artifacts
        metadata = build_artifacts(data_path=source, output=output, queries=queries)
        metadata["preparation"] = signature
        temporary = output / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "metadata.json")
        marker.unlink()
        return metadata

def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="CSV, ZIP hoặc thư mục CSV PRSA; mặc định ưu tiên ../data")
    parser.add_argument("--output", help="Thư mục artifact; mặc định theo vị trí mã nguồn")
    parser.add_argument("--rebuild", action="store_true", help="Buộc tạo lại artifact")
    parser.add_argument("--queries", type=int, default=120)
    args = parser.parse_args()
    try:
        prepare(args.data, args.output, args.rebuild, args.queries)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Không chuẩn bị được module 3: {exc}\n")

if __name__ == "__main__":
    main()
