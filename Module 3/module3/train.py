"""Chuẩn bị artifact tái lập được cho module 3 (mặc định đủ 12 trạm)."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import sklearn
from .data import FEATURES, TRAIN_END, VALID_END, SOURCE_URL, read_data, prepare_data, split_by_time
from .similarity import SimilaritySearch
from .paths import DEFAULT_ARTIFACTS, default_data_source, source_fingerprint

SCOPES = ["same_station", "all_stations"]

def sample_queries(queries, limit=120, seed=42):
    """Lấy mẫu phân bổ đều theo trạm; cùng mẫu cho K và metric khác nhau."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or queries.empty:
        raise ValueError("Cần tập truy vấn không rỗng và limit nguyên dương.")
    queries = queries.reset_index(drop=True)
    groups = list(queries.groupby("station", sort=True))
    chunks = []
    target = min(limit, len(queries))
    quota, remainder = divmod(target, len(groups))
    for i, (_, group) in enumerate(groups):
        n = min(len(group), quota + (i < remainder))
        if n:
            chunks.append(group.sample(n, random_state=seed))
    result = pd.concat(chunks) if chunks else queries.iloc[:0]
    if len(result) < target:
        remaining = queries.drop(index=result.index)
        result = pd.concat([result, remaining.sample(target-len(result),random_state=seed)])
    return result.reset_index(drop=True)

def evaluate(searcher, queries, k, limit=120, scope="same_station", return_details=False):
    if scope not in SCOPES:
        raise ValueError(f"scope phải thuộc {SCOPES}")
    sample = sample_queries(queries, limit)
    records = []
    rng = np.random.default_rng(42)
    for _, query in sample.iterrows():
        station = None if scope == "same_station" else "ALL"
        started = perf_counter()
        result = searcher.find_similar_periods(query, k=k, station=station)
        elapsed = (perf_counter()-started)*1000
        q = searcher.scaler.transform(pd.DataFrame([query[searcher.features].to_dict()]))[0]
        candidates = searcher.station_indices[str(query["station"])][0] if scope == "same_station" else np.arange(len(searcher.rows))
        selected = rng.choice(candidates, size=k, replace=False)
        delta = np.abs(searcher.matrix[selected]-q)
        baseline = np.sqrt((delta**2).sum(axis=1)) if searcher.metric == "euclidean" else delta.sum(axis=1)
        records.append({"station": str(query["station"]), "datetime": str(query["datetime"]), "metric": searcher.metric, "scope": scope, "k": int(k), "mean_distance": float(result.distance.mean()), "random_mean_distance": float(baseline.mean()), "query_ms": elapsed, **{f"mean_abs_gap_{f}": float(np.abs(result[f].to_numpy()-float(query[f])).mean()) for f in searcher.features}})
    details = pd.DataFrame(records)
    summary = {"metric": searcher.metric, "scope": scope, "k": int(k), "queries": len(details), "query_stations": int(details.station.nunique()), "mean_distance": float(details.mean_distance.mean()), "random_mean_distance": float(details.random_mean_distance.mean()), "mean_query_ms": float(details.query_ms.mean()), "p95_query_ms": float(details.query_ms.quantile(.95)), **{c: float(details[c].mean()) for c in details if c.startswith("mean_abs_gap_")}}
    return (summary, details) if return_details else summary

def build_artifacts(data_path=None, output=None, station="ALL", train_end=TRAIN_END, valid_end=VALID_END, metric="euclidean", queries=120):
    if metric not in {"euclidean", "manhattan"}:
        raise ValueError("Metric không hợp lệ.")
    source = Path(data_path) if data_path is not None else default_data_source()
    raw = read_data(source)
    df, quality = prepare_data(raw, station=station)
    train, valid, test = split_by_time(df, train_end, valid_end)
    stations = sorted(df.station.unique().tolist())
    for name, part in [("train",train),("validation",valid),("test",test)]:
        absent = set(stations)-set(part.station.unique())
        if part.empty or absent:
            raise ValueError(f"Tập {name} rỗng hoặc thiếu trạm {sorted(absent)}. Điều chỉnh mốc chia hoặc phạm vi trạm.")
    if train.groupby("station").size().min() < 10:
        raise ValueError("Mỗi trạm cần ít nhất 10 quan sát train để khảo sát K=3/5/10.")
    # Validate sampling configuration before writing output.
    sample_queries(valid, queries)
    output = Path(output) if output is not None else DEFAULT_ARTIFACTS
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(quality["by_station"]).to_csv(output/"data_quality_by_station.csv",index=False)
    split_rows = []
    for split, part in [("train",train),("validation",valid),("test",test)]:
        for name, group in part.groupby("station",sort=True):
            split_rows.append({"split":split,"station":name,"rows":len(group),"start":str(group.datetime.min()),"end":str(group.datetime.max())})
    pd.DataFrame(split_rows).to_csv(output/"split_summary.csv",index=False)
    validation_rows = []
    model_files = {}
    active_model = None
    for distance in ["euclidean","manhattan"]:
        print(f"Xây kho {distance}: {len(train):,} bản ghi / {len(stations)} trạm",flush=True)
        model = SimilaritySearch(metric=distance).fit(train)
        filename = "knn.joblib" if distance == metric else f"knn_{distance}.joblib"
        model.save(output/filename)
        model_files[distance] = filename
        for scope in SCOPES:
            for k in [3,5,10]:
                validation_rows.append(evaluate(model, valid, k, queries, scope))
        if distance == metric:
            active_model = model
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output/"validation_metrics.csv",index=False)
    # Cấu hình metric/K đã được đặt trước, test không dùng để chọn mô hình.
    test_rows, detail_rows = [], []
    for scope in SCOPES:
        summary, detail = evaluate(active_model,test,5,queries,scope,return_details=True)
        test_rows.append(summary)
        detail_rows.append(detail)
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(output/"test_metrics.csv",index=False)
    details = pd.concat(detail_rows,ignore_index=True)
    details.to_csv(output/"test_query_details.csv",index=False)
    per_station = details.groupby(["station","metric","scope","k"],as_index=False).agg(queries=("datetime","size"),mean_distance=("mean_distance","mean"),random_mean_distance=("random_mean_distance","mean"),mean_query_ms=("query_ms","mean"))
    per_station.to_csv(output/"test_metrics_by_station.csv",index=False)
    valid.to_csv(output/"validation.csv",index=False)
    test.to_csv(output/"test.csv",index=False)
    examples = []
    query_rows = []
    for name, group in test.groupby("station",sort=True):
        query = group.iloc[0]
        query_rows.append(query.to_dict())
        for scope in SCOPES:
            result = active_model.find_similar_periods(query,k=5,station="ALL" if scope=="all_stations" else None)
            result.insert(0,"scope",scope)
            result.insert(0,"query_station",name)
            result.insert(0,"query_datetime",str(query.datetime))
            examples.append(result)
    pd.concat(examples,ignore_index=True).to_csv(output/"example_neighbors.csv",index=False)
    pd.DataFrame(query_rows).to_csv(output/"example_queries.csv",index=False)
    checksum = source_fingerprint(source)
    metadata = {**quality,"artifact_version":2,"station":station,"features":FEATURES,"train_end":str(train_end),"valid_end":str(valid_end),"train_rows":len(train),"validation_rows":len(valid),"test_rows":len(test),"reference_max_time":str(active_model.reference_max_time),"metric":metric,"k":5,"model_files":model_files,"queries_per_evaluation":min(queries,len(test)),"sampling":"station-balanced, random_state=42","missing_policy":"drop missing/invalid selected features; no imputation","timezone":"Asia/Shanghai, stored as naive local datetime","source":SOURCE_URL,"source_sha256":checksum,"versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scikit-learn":sklearn.__version__}}
    (output/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    from .reporting import write_report
    write_report(output, metadata)
    print(f"Hoàn tất: {len(stations)} trạm; train={len(train):,}, validation={len(valid):,}, test={len(test):,}",flush=True)
    return metadata

def main():
    import sys
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default=None,help="CSV/ZIP/thư mục PRSA; mặc định ưu tiên data chung trong repo")
    parser.add_argument("--station",default="ALL",help="ALL (mặc định) hoặc tên trạm")
    parser.add_argument("--output",default=None,help="Mặc định artifacts/module3 theo vị trí mã nguồn")
    parser.add_argument("--train-end",default=TRAIN_END)
    parser.add_argument("--valid-end",default=VALID_END)
    parser.add_argument("--metric",choices=["euclidean","manhattan"],default="euclidean")
    parser.add_argument("--queries",type=int,default=120)
    args=parser.parse_args()
    if args.queries<1:
        parser.error("--queries phải dương")
    build_artifacts(data_path=args.data,output=args.output,station=args.station,train_end=args.train_end,valid_end=args.valid_end,metric=args.metric,queries=args.queries)

if __name__=="__main__":
    main()
