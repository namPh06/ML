"""Tạo biểu đồ và báo cáo thực nghiệm từ các CSV đã sinh."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def markdown_table(df):
    frame = df.copy()
    for c in frame.select_dtypes(include="number"):
        frame[c] = frame[c].map(lambda x: f"{x:.4f}" if isinstance(x,(float,np.floating)) else str(x))
    rows = ["| " + " | ".join(map(str,frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(str(x).replace("|","/") for x in row) + " |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(rows)

def write_report(output, metadata=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import FEATURES, UNITS
    output=Path(output)
    if metadata is None:
        metadata=json.loads((output/"metadata.json").read_text(encoding="utf-8"))
    validation=pd.read_csv(output/"validation_metrics.csv")
    test=pd.read_csv(output/"test_metrics.csv")
    split=pd.read_csv(output/"split_summary.csv")
    quality=pd.read_csv(output/"data_quality_by_station.csv")
    counts=split.pivot(index="station",columns="split",values="rows")[["train","validation","test"]]
    ax=counts.plot.bar(stacked=True,figsize=(12,5),color=["#246c91","#5faea6","#c1d9de"])
    ax.set(xlabel="Station",ylabel="Complete observations",title="Data split by station")
    plt.xticks(rotation=35,ha="right");plt.tight_layout()
    plt.savefig(output/"station_coverage.png",dpi=150);plt.close()
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,metric in enumerate(["euclidean","manhattan"]):
        for j,scope in enumerate(["same_station","all_stations"]):
            part=validation[(validation.metric==metric)&(validation.scope==scope)].sort_values("k")
            ax=axes[i,j]
            ax.plot(part.k,part.mean_distance,"o-",label="KNN")
            ax.plot(part.k,part.random_mean_distance,"o--",label="Random baseline")
            ax.set(title=f"{metric} / {scope}",xlabel="K",ylabel="Mean distance")
            ax.set_xticks([3,5,10]);ax.legend()
    fig.tight_layout();fig.savefig(output/"validation_comparison.png",dpi=150);plt.close(fig)
    examples=pd.read_csv(output/"example_neighbors.csv")
    queries=pd.read_csv(output/"example_queries.csv")
    query=queries.iloc[0]
    neighbors=examples[(examples.query_station==query.station)&(examples.scope=="all_stations")]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,feature in zip(axes.ravel(),FEATURES):
        delta=neighbors[feature].to_numpy()-query[feature]
        ax.barh([f"N{rank}" for rank in neighbors["rank"]],delta,color="#246c91")
        ax.axvline(0,color="#555555",linewidth=.8)
        ax.set_title(f"{feature}: query={query[feature]:.2f}")
        ax.set_xlabel(f"Neighbor minus query ({UNITS[feature]})")
        ax.invert_yaxis()
    fig.suptitle(f"Example query: {query.station}, {query.datetime}")
    fig.tight_layout();fig.savefig(output/"example_comparison.png",dpi=150);plt.close(fig)
    selected=test[["scope","queries","mean_distance","random_mean_distance","mean_query_ms","p95_query_ms"]]
    text=f"""# Báo cáo module 3 tìm kiếm tương đồng

Dữ liệu: UCI Beijing Multi-Site Air Quality, {len(metadata['stations'])} trạm. Module tìm các quan sát theo giờ có tổ hợp đặc trưng gần truy vấn. Không thực hiện dự báo PM2.5 hoặc phân loại AQI.

## Dữ liệu và phương pháp

Nguồn: {metadata['source']}. DOI 10.24432/C5RK5G, CC BY 4.0. Dữ liệu quan trắc ô nhiễm từ Beijing Municipal Environmental Monitoring Center, khí tượng từ China Meteorological Administration. Múi giờ lưu trữ: {metadata['timezone']}.

Tổng dòng đầu vào: {metadata['input_rows']:,}. Dòng sử dụng: {metadata['rows_used']:,}. Bỏ do thiếu/không hợp lệ trong đặc trưng: {metadata['rows_dropped_missing']:,}. Loại hàng thiếu có thể tạo sai lệch mẫu.

Đặc trưng: {', '.join(metadata['features'])}. StandardScaler chỉ học trên train; dùng cùng scaler toàn kho khi tìm trong một trạm hoặc xuyên trạm. Không đưa mã trạm hay timestamp vào khoảng cách. KNN tạo sẵn chỉ mục toàn kho và từng trạm.

Train trước {metadata['train_end']}; validation từ mốc đó đến trước {metadata['valid_end']}; test từ mốc sau trở đi. Kho train kết thúc {metadata['reference_max_time']}. Không dùng quan sát validation/test làm hàng xóm.

## Số lượng theo trạm

{markdown_table(counts.reset_index())}

## Chất lượng dữ liệu theo trạm

{markdown_table(quality[['station','rows_before_missing_filter','rows_used','rows_dropped_missing']])}

## Khảo sát validation

Euclidean/Manhattan × K=3/5/10 × cùng trạm/toàn kho. Mẫu truy vấn cân bằng theo trạm, random_state=42. Đối chứng ngẫu nhiên lấy cùng số hàng xóm trong đúng phạm vi tìm kiếm. Không so trực tiếp khoảng cách của các metric khác nhau, không tự chọn cấu hình chỉ dựa trên khoảng cách nhỏ.

{markdown_table(validation[['metric','scope','k','queries','mean_distance','random_mean_distance','mean_query_ms']])}

## Đánh giá test

Cấu hình được đặt trước: {metadata['metric']}, K={metadata['k']}. Test chỉ báo cáo cấu hình này với hai phạm vi. Không có lựa chọn cấu hình theo kết quả test. Kết quả từng trạm nằm trong test_metrics_by_station.csv; chênh lệch từng đặc trưng theo đơn vị gốc nằm trong test_metrics.csv và test_query_details.csv.

{markdown_table(selected)}

Thời gian đo phụ thuộc máy và tải hệ thống, chưa phải benchmark dịch vụ. Số truy vấn được lấy mẫu không phải toàn bộ test.

## Diễn giải và giới hạn

Khoảng cách nhỏ hơn đối chứng cho thấy KNN tìm được các vector gần hơn theo đặc trưng đã chọn; đây không phải Accuracy hoặc bằng chứng dự báo chính xác. PM2.5 là đầu vào, nên PM2.5 gần nhau là điều được kỳ vọng. Không suy ra phần trăm tương đồng. KNN vẫn trả K hàng xóm nếu đủ mẫu, ngay cả khi truy vấn nằm xa kho. Các giờ liền nhau có thể cùng xuất hiện; chưa tìm cả chuỗi nhiều giờ. Khác trạm không bảo đảm bối cảnh địa lý hoặc nguồn phát thải giống nhau. Đánh giá thủ công bảng hàng xóm và từng biến vẫn cần thiết.

Giao diện cho nhập điều kiện hoặc chọn thời điểm sau train, lọc một trạm hoặc toàn bộ trạm, chọn metric/K, hiển thị bảng và tải CSV. Artifact chứa scaler, đặc trưng, các chỉ mục và bảng gốc; không được đổi thứ tự bảng riêng khỏi mô hình.

## Biểu đồ

- station_coverage.png: phân bố dữ liệu theo trạm và tập.
- validation_comparison.png: so KNN với đối chứng, tách metric và phạm vi.
- example_comparison.png: chênh lệch từng biến của 5 hàng xóm với truy vấn.

Báo cáo sinh tự động từ các CSV khi chạy python -m module3.train. Không nhận xét chung rằng một metric tốt nhất chỉ từ các bảng này.
"""
    (output/"REPORT_MODULE3.md").write_text(text,encoding="utf-8")
