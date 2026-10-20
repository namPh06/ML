"""Giao diện độc lập hoặc nhập render() vào ứng dụng chung của nhóm."""
from pathlib import Path
from time import perf_counter
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import pandas as pd
import streamlit as st
from module3.data import UNITS
from module3.similarity import SimilaritySearch

@st.cache_resource
def load_model(filename, modified):
    return SimilaritySearch.load(filename)

@st.cache_data
def load_table(filename, modified):
    frame=pd.read_csv(filename)
    if "datetime" in frame:
        frame["datetime"]=pd.to_datetime(frame["datetime"])
    return frame

def read_table(file):
    return load_table(str(file),file.stat().st_mtime_ns)


FEATURE_NOTES = {
    "PM2.5": ("Bụi mịn PM2.5", "Hạt bụi có đường kính khí động học ≤ 2,5 µm; có thể đi sâu vào phổi. Số càng lớn, lượng bụi trong không khí càng nhiều."),
    "PM10": ("Bụi PM10", "Hạt bụi có đường kính khí động học ≤ 10 µm, bao gồm cả PM2.5. Không cộng hai cột này để tính tổng bụi."),
    "TEMP": ("Nhiệt độ không khí", "Đơn vị °C. Giá trị thấp hơn nghĩa là lạnh hơn; giá trị âm là dưới 0 °C. Không phải nồng độ ô nhiễm."),
    "PRES": ("Áp suất khí quyển", "Đơn vị hPa. Số lớn hơn là áp suất cao hơn; còn phụ thuộc độ cao và thời tiết. Không phải mức ô nhiễm."),
    "DEWP": ("Nhiệt độ điểm sương", "Nhiệt độ không khí cần hạ đến để hơi nước bắt đầu ngưng tụ. Cùng nhiệt độ không khí, điểm sương càng gần TEMP thì độ ẩm tương đối càng cao. Không phải độ ẩm %; có thể âm."),
    "WSPM": ("Tốc độ gió", "Đơn vị m/s; 1 m/s = 3,6 km/h. Số nhỏ là gió yếu, số lớn là gió mạnh. Gió có thể phân tán hoặc mang bụi tới; không tự quyết định chất lượng không khí."),
}
WHO_SOURCE = "https://www.who.int/publications/i/item/9789240034228"

def observation_columns(features):
    config = {
        "datetime": st.column_config.DatetimeColumn("Thời điểm (UTC+8)", format="DD/MM/YYYY HH:mm", help="Giờ địa phương Bắc Kinh, sớm hơn giờ Việt Nam 1 giờ."),
        "station": st.column_config.TextColumn("Trạm", help="Tên trạm quan trắc tại Bắc Kinh."),
        "rank": st.column_config.NumberColumn("Hạng", help="Xếp theo khoảng cách tăng dần; hạng 1 gần truy vấn nhất.", format="%d"),
        "distance": st.column_config.NumberColumn("Khoảng cách", help="Khoảng cách trên các biến đã chuẩn hóa. Càng nhỏ càng gần truy vấn; không có đơn vị, không phải phần trăm hoặc AQI. Chỉ so trong cùng metric.", format="%.4f"),
    }
    for feature in features:
        name, explanation = FEATURE_NOTES.get(feature, (feature, ""))
        config[feature] = st.column_config.NumberColumn(f"{feature} · {name} ({UNITS.get(feature, '')})", help=explanation, format="%.2f")
    return config

@st.cache_data
def historical_bands(filename, modified, station):
    rows = load_model(filename, modified).rows
    if station != "ALL":
        rows = rows[rows.station.eq(station)]
    return rows[list(FEATURE_NOTES)].quantile([0.25, 0.75])

def render_feature_notes(bands, scope_name):
    st.markdown("**Chú giải các cột và cách đọc giá trị**")
    notes = []
    for feature, (name, explanation) in FEATURE_NOTES.items():
        unit = UNITS[feature]
        low, high = bands.loc[0.25, feature], bands.loc[0.75, feature]
        notes.append({"Ký hiệu": feature, "Ý nghĩa": name, "Đơn vị": unit,
                      "Thấp trong kho": f"≤ {low:.2f}", "Cao trong kho": f"≥ {high:.2f}",
                      "Cách hiểu": explanation})
    st.dataframe(pd.DataFrame(notes), hide_index=True, width="stretch")
    st.caption(f"Mức thấp/cao trên được tính từ kho train ({scope_name}): thấp là nhóm 25% giá trị nhỏ nhất (Q25), cao là nhóm 25% lớn nhất (Q75); ở giữa là mức trung gian. Đây là so sánh thống kê trong bộ dữ liệu Bắc Kinh, không phải ngưỡng sức khỏe hay nhãn AQI. µg/m³ = microgam bụi trong một mét khối không khí; µm = micromét kích thước hạt.")
    st.info("Với PM2.5 và PM10, nồng độ càng cao nghĩa là không khí chứa càng nhiều bụi. Nồng độ thấp hơn thường có lợi hơn cho sức khỏe, nhưng không có nghĩa là hoàn toàn không có rủi ro. TEMP, PRES, DEWP và WSPM là điều kiện thời tiết; cao/thấp không đồng nghĩa với tốt/xấu.")
    with st.expander("Bao nhiêu bụi là cao? Xem mốc tham khảo WHO và ví dụ", expanded=True):
        st.markdown("**Mốc WHO 2021 (µg/m³)** — áp dụng cho nồng độ lấy trung bình đúng khoảng thời gian ghi dưới đây.")
        st.dataframe(pd.DataFrame([
            {"Chỉ số": "PM2.5", "Trung bình 24 giờ": 15, "Trung bình năm": 5},
            {"Chỉ số": "PM10", "Trung bình 24 giờ": 45, "Trung bình năm": 15},
        ]), hide_index=True)
        st.markdown("Ví dụ: nếu **trung bình đủ 24 giờ** của PM2.5 là 30 µg/m³ thì bằng 2 lần mốc 15; PM10 là 90 µg/m³ thì bằng 2 lần mốc 45. Trên mốc nghĩa là vượt mức hướng dẫn tương ứng; dưới mốc không bảo đảm không có rủi ro.")
        st.warning("Mỗi hàng trong bảng truy vấn/kết quả là quan sát theo giờ. Không dùng trực tiếp mốc 24 giờ hoặc mốc năm để kết luận hàng đó ‘an toàn’, ‘nguy hại’ hoặc đạt/không đạt tiêu chuẩn. Ứng dụng chưa tính trung bình 24 giờ hay AQI; các mức thấp/cao trong kho chỉ giúp đọc dữ liệu.")
        st.caption("Mốc 24 giờ của WHO gắn với phân vị 99 của các trung bình ngày trong năm (khoảng 3–4 ngày vượt/năm); đây là hướng dẫn bảo vệ sức khỏe, không phải tiêu chuẩn pháp lý Việt Nam.")
        st.markdown(f"Nguồn: [WHO Global Air Quality Guidelines 2021]({WHO_SOURCE}).")

def observation_caption():
    st.caption("PM2.5/PM10: nồng độ bụi (µg/m³) · TEMP: nhiệt độ (°C) · PRES: áp suất (hPa) · DEWP: điểm sương (°C) · WSPM: tốc độ gió (m/s). Di chuột lên tiêu đề cột để xem giải thích. Thấp/cao trong kho không phải nhãn sức khỏe.")

def render(artifacts=None):
    artifacts=Path(artifacts) if artifacts else ROOT/"artifacts/module3"
    st.title("Tìm thời điểm có điều kiện không khí tương đồng")
    st.write("Tra cứu các quan sát lịch sử gần với thời điểm hoặc điều kiện bạn chọn.")
    metadata_file=artifacts/"metadata.json"
    if not metadata_file.exists():
        st.info("Chưa có kho lịch sử. Từ thư mục repo (có run_module3.py), chạy: python run_module3.py --prepare-only. Sau đó tải lại trang.")
        return
    metadata=json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("artifact_version") != 2:
        st.info("Kho lịch sử cần cập nhật. Chạy python run_module3.py --rebuild từ thư mục repo để tạo lại bộ tìm kiếm.")
        return
    metric=st.sidebar.selectbox("Khoảng cách",["euclidean","manhattan"],index=0 if metadata["metric"]=="euclidean" else 1,key="m3_metric")
    filename=metadata["model_files"][metric]
    model_path=artifacts/filename
    if not model_path.exists():
        st.error("Thiếu tệp bộ tìm kiếm. Hãy chạy lại lệnh chuẩn bị trong README.")
        return
    try:
        model=load_model(str(model_path),model_path.stat().st_mtime_ns)
    except (ValueError,OSError) as exc:
        st.error(str(exc));return
    stations=sorted(model.station_indices)
    scope=st.sidebar.radio("Phạm vi tìm kiếm",["Một trạm","Toàn bộ các trạm"],key="m3_scope")
    station=st.sidebar.selectbox("Trạm truy vấn / trạm mẫu",stations,key="m3_station")
    st.sidebar.caption("Với thời điểm có sẵn, đây là trạm của truy vấn. Khi nhập tay, trạm này cung cấp giá trị mẫu.")
    available=len(model.rows) if scope=="Toàn bộ các trạm" else len(model.station_indices[station][0])
    k=int(st.sidebar.number_input("Số kết quả K",min_value=1,max_value=min(50,available),value=min(5,available),step=1,key="m3_k"))
    cols=st.columns(3)
    cols[0].metric("Trạm trong kho",len(stations))
    cols[1].metric("Quan sát lịch sử",f"{len(model.rows):,}")
    cols[2].metric("Ứng viên đang tìm",f"{available:,}")
    st.caption(f"Kho kết thúc {model.reference_max_time:%d/%m/%Y %H:%M}, giờ Bắc Kinh (UTC+8). Truy vấn có sẵn nằm sau kho lịch sử.")
    search_tab,data_tab,evaluation_tab,guide_tab=st.tabs(["Tìm kiếm","Dữ liệu","Đánh giá","Cách hiểu kết quả"])
    with search_tab:
        bands=historical_bands(str(model_path),model_path.stat().st_mtime_ns,"ALL" if scope=="Toàn bộ các trạm" else station)
        render_feature_notes(bands,"toàn bộ các trạm" if scope=="Toàn bộ các trạm" else f"trạm {station}")
        mode=st.radio("Cách truy vấn",["Chọn thời điểm","Nhập điều kiện"],horizontal=True,key="m3_mode")
        if mode=="Chọn thời điểm":
            query_file=artifacts/"test.csv"
            if not query_file.exists():
                st.error("Thiếu dữ liệu thời điểm truy vấn. Hãy chạy lại lệnh chuẩn bị.");return
            queries=read_table(query_file)
            queries=queries[queries.station.eq(station)].sort_values("datetime")
            if queries.empty:
                st.warning("Không có thời điểm truy vấn cho trạm này.");return
            days=sorted(queries.datetime.dt.date.unique())
            day=st.selectbox("Ngày truy vấn",days,format_func=lambda d:d.strftime("%d/%m/%Y"),key="m3_day")
            daily=queries[queries.datetime.dt.date.eq(day)].reset_index(drop=True)
            selected=st.selectbox("Giờ truy vấn",range(len(daily)),format_func=lambda i:daily.iloc[i].datetime.strftime("%H:%M"),key="m3_hour")
            query=daily.iloc[selected].to_dict()
            st.markdown("**Điều kiện tại thời điểm truy vấn**")
            observation_caption()
            st.dataframe(pd.DataFrame([query])[["datetime","station",*model.features]],hide_index=True,column_config=observation_columns(model.features),width="stretch")
        else:
            query={"station":station}
            defaults=model.rows.iloc[model.station_indices[station][0]][model.features].median()
            cols=st.columns(3)
            for i,feature in enumerate(model.features):
                with cols[i%3]:
                    kwargs={"min_value":0.0} if feature in {"PM2.5","PM10","WSPM","RAIN"} else ({"min_value":0.01} if feature=="PRES" else {})
                    query[feature]=st.number_input(f"{feature} ({UNITS.get(feature,'')})",value=float(defaults[feature]),format="%.2f",key=f"m3_input_{station}_{feature}",help=FEATURE_NOTES.get(feature,(feature,""))[1],**kwargs)
            st.caption("Bạn có thể sửa các giá trị mẫu. Điều kiện nhập được đối chiếu với kho lịch sử cố định.")
        target="ALL" if scope=="Toàn bộ các trạm" else station
        signature=json.dumps({"query":query,"scope":target,"k":k,"metric":metric,"artifact":model_path.stat().st_mtime_ns},default=str,sort_keys=True)
        if st.button("Tìm thời điểm tương đồng",type="primary",key="m3_search"):
            try:
                started=perf_counter()
                result=model.find_similar_periods(query,k=k,station=target)
                st.session_state["m3_result"]={"signature":signature,"query":query,"result":result,"ms":(perf_counter()-started)*1000}
            except (ValueError,TypeError) as exc:
                st.session_state.pop("m3_result",None)
                st.error(str(exc))
        saved=st.session_state.get("m3_result")
        if saved and saved["signature"]==signature:
            result=saved["result"]
            st.subheader("Các thời điểm gần nhất")
            st.caption(f"Tìm được {len(result)} kết quả trong {saved['ms']:.1f} ms. Phạm vi: {scope.lower()}.")
            observation_caption()
            st.caption("Hạng 1 là kết quả gần nhất. Khoảng cách tính trên các biến đã chuẩn hóa, càng nhỏ càng giống điều kiện truy vấn; không đo mức ô nhiễm.")
            st.dataframe(result,hide_index=True,width="stretch",column_config=observation_columns(model.features))
            st.download_button("Tải kết quả CSV",result.to_csv(index=False).encode("utf-8-sig"),"similar_periods.csv","text/csv",key="m3_download")
            st.caption("Khoảng cách càng nhỏ càng gần theo sáu biến đã chọn. Đây không phải phần trăm tương đồng; kể cả kết quả gần nhất cũng có thể khác đáng kể.")
            st.subheader("Chênh lệch so với truy vấn")
            gaps=result[["rank","station","datetime"]].copy()
            for feature in model.features:
                gaps[f"Δ {feature} ({UNITS.get(feature,'')})"]=result[feature]-float(query[feature])
            st.caption("Δ = giá trị kết quả − giá trị truy vấn, giữ nguyên đơn vị gốc: Δ dương là cao hơn truy vấn, Δ âm là thấp hơn, 0 là bằng nhau. Δ không phải mức tăng/giảm theo thời gian và không phải phần trăm.")
            observation_caption()
            st.dataframe(gaps,hide_index=True,width="stretch",column_config=observation_columns([]))
            with st.expander("Biểu đồ so sánh từng điều kiện",expanded=True):
                cols=st.columns(2)
                for i,feature in enumerate(model.features):
                    chart=pd.DataFrame({feature:[float(query[feature]),*result[feature].tolist()]},index=["Truy vấn",*[f"Kết quả {rank}" for rank in result['rank']]])
                    with cols[i%2]:
                        st.caption(f"{feature} ({UNITS.get(feature,'')})")
                        st.bar_chart(chart)
        elif saved:
            st.info("Điều kiện đã thay đổi. Bấm tìm để cập nhật kết quả.")
    with data_tab:
        st.subheader("Nguồn và dữ liệu đang dùng")
        st.write("UCI Beijing Multi-Site Air Quality: ô nhiễm từ Beijing Municipal Environmental Monitoring Center, khí tượng từ China Meteorological Administration.")
        st.link_button("Xem dataset UCI",metadata["source"])
        st.write(f"Có {metadata['rows_used']:,} quan sát đủ đặc trưng; {metadata['rows_dropped_missing']:,} quan sát bị loại do thiếu hoặc không hợp lệ. Không điền bằng dữ liệu tương lai.")
        for filename,title in [("data_quality_by_station.csv","Chất lượng theo trạm"),("split_summary.csv","Chia dữ liệu theo thời gian")]:
            file=artifacts/filename
            if file.exists():
                st.write(title);st.dataframe(read_table(file),hide_index=True)
        if (artifacts/"station_coverage.png").exists():
            st.image(str(artifacts/"station_coverage.png"))
    with evaluation_tab:
        st.subheader("Thực nghiệm tìm kiếm tương đồng")
        st.write("Validation khảo sát hai khoảng cách và K=3/5/10. Test báo cáo cấu hình đã đặt trước; không dùng Accuracy cho chức năng truy xuất này.")
        for filename,title in [("validation_metrics.csv","Khảo sát validation"),("test_metrics.csv","Cấu hình cuối trên test"),("test_metrics_by_station.csv","Test theo từng trạm")]:
            file=artifacts/filename
            if file.exists():
                st.write(title);st.dataframe(read_table(file),hide_index=True)
        st.caption("same_station: cùng trạm; all_stations: toàn kho. Random baseline chọn ngẫu nhiên trong đúng phạm vi. Không so trực tiếp trị số khoảng cách giữa hai metric. Thời gian phụ thuộc máy.")
        report=artifacts/"REPORT_MODULE3.md"
        if report.exists():
            st.download_button("Tải báo cáo thực nghiệm",report.read_bytes(),"REPORT_MODULE3.md","text/markdown",key="m3_report")
    with guide_tab:
        st.markdown("""**Cách dùng:** chọn trạm của truy vấn, phạm vi tìm kiếm, khoảng cách và K. Sau đó chọn thời điểm hoặc nhập sáu điều kiện rồi bấm tìm.

**Các biến:** PM2.5 và PM10 là nồng độ bụi; TEMP là nhiệt độ; PRES là áp suất; DEWP là nhiệt độ điểm sương; WSPM là tốc độ gió. Điểm sương không phải độ ẩm tương đối.

**Cách tính:** chuẩn hóa bằng thông số học từ kho train, rồi tìm các vector gần nhất. Trạm và thời gian chỉ để xác định bản ghi, không được mã hóa thành số để tính khoảng cách.

**Giới hạn:** đây là các quan sát theo từng giờ; các giờ liền nhau có thể cùng xuất hiện. Đặc trưng gần nhau không bảo đảm nguồn ô nhiễm giống nhau. Module chưa tìm cả chuỗi nhiều giờ và chưa đưa ra dự báo hay nhãn AQI.

**Dữ liệu Bắc Kinh:** các mốc giờ đang dùng là UTC+8, không phải giờ Việt Nam. Không mặc nhiên áp dụng kết quả cho Việt Nam.""")

if __name__=="__main__":
    st.set_page_config(page_title="Tìm thời điểm tương đồng",page_icon="🌤️",layout="wide")
    render()
