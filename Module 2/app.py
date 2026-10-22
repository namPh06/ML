"""Interactive dashboard for Module 2's data, model and historical forecasts."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from module2 import ROOT, ARTIFACTS, REPORTS, CLASS_NAMES, NUMERIC, load_data, build_features, predict_pollution_class, labels

st.set_page_config(page_title='Air Quality Lab · Module 2', page_icon='🌫️', layout='wide')
st.markdown('''<style>
.stApp {background:#f5f9fc;color:#18354b;}
[data-testid="stHeader"] {background:#f5f9fc;}
.block-container {width:100%;max-width:none;box-sizing:border-box;padding:4rem clamp(1rem,3vw,3rem) 3rem;}
[data-testid="stColumn"], [data-testid="stVerticalBlock"] {min-width:0;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#e4f2f4,#edf3fa);border-right:1px solid #c9dfe6;}
[data-testid="stSidebar"] h1 {color:#126779;font-size:1.55rem;letter-spacing:-.5px;}
[data-testid="stSidebar"] label {padding:5px 8px;border-radius:8px;}
[data-testid="stSidebar"] label:has(input:checked) {background:#cde8ed;box-shadow:inset 3px 0 #167d8d;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#fff,#eaf5f8);border:1px solid #c6e0e8;border-top:3px solid #228b9c;border-radius:12px;padding:14px;height:100%;}
[data-testid="stMetricValue"] {font-size:clamp(1.25rem,2vw,2rem);color:#135d73;}
[data-testid="stMetricValue"] div, [data-testid="stMetricValue"] p,
[data-testid="stMetricLabel"] p, [data-testid="stMetricDelta"] p {
white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere;}
[data-testid="stCaptionContainer"] p {color:#526b7b;overflow-wrap:anywhere;}
[data-testid="stMarkdownContainer"] p {overflow-wrap:anywhere;}
[data-testid="stMain"], [data-testid="stSidebarContent"] {overflow-y:auto;scrollbar-gutter:stable;}
* {scrollbar-width:auto;scrollbar-color:#709aa8 #eaf2f6;}
*::-webkit-scrollbar {width:12px;height:12px;}
*::-webkit-scrollbar-thumb {background:#709aa8;border:2px solid #eaf2f6;border-radius:8px;}
*::-webkit-scrollbar-track {background:#eaf2f6;}
[data-testid="stExpander"] {background:#fff;border:1px solid #cee0e8;border-radius:12px;}
[data-testid="stExpander"] summary {background:#eaf3f8;border-radius:12px;color:#195c74;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#cee0e8;}
button[kind="primary"] {background:#167d8d;border-color:#167d8d;color:#fff;}
button[kind="primary"]:hover {background:#106472;border-color:#106472;color:#fff;}
button[kind="secondary"] {border-color:#bdd8e2;}
h1 {color:#164e67;letter-spacing:-1px;} h2,h3 {color:#1b6578;letter-spacing:-.4px;}
@media(max-width:900px) {
  .block-container {padding-left:1rem;padding-right:1rem;}
  [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 240px;min-width:0;}
}
</style>''', unsafe_allow_html=True)


@st.cache_data
def read_csv(path, modified):
    return pd.read_csv(path)


def table(name):
    path = REPORTS/name
    return read_csv(str(path), path.stat().st_mtime_ns)


@st.cache_data
def history(signature):
    return load_data()[0]


def get_history():
    paths = sorted((ROOT.parent/'data').glob('PRSA_Data_*.csv'))
    return history(tuple((p.name, p.stat().st_mtime_ns) for p in paths))


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def download_csv(frame, name):
    st.download_button('Tải bảng CSV', frame.to_csv(index=False).encode('utf-8-sig'), name, 'text/csv')


def show_table(frame, **kwargs):
    """Keep short tables expanded and use native virtual scrolling for long tables."""
    if len(frame) > 14:
        st.caption(f'Đầy đủ {len(frame):,} hàng. Đặt con trỏ trong bảng để cuộn dọc; kéo thanh cuộn dưới bảng để xem các cột bên phải. Đưa con trỏ ra ngoài bảng để cuộn tiếp trang.')
    st.dataframe(frame, height=min(560, (len(frame)+1)*36+4), row_height=36,
                 column_config={name: st.column_config.Column(width='medium') for name in frame.columns}, **kwargs)


required = [ARTIFACTS/'metadata.json', ARTIFACTS/'model.joblib'] + [REPORTS/name for name in [
    'metrics.csv', 'data_audit.json', 'classification_report.json', 'split_support.csv',
    'experiments.csv', 'test_slices.csv', 'test_predictions.csv', 'coefficients.csv', 'missing_by_station.csv']]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error('Chưa đủ kết quả dự án: ' + ', '.join(missing))
    st.code('python train.py', language='powershell')
    st.stop()

metadata = load_json(ARTIFACTS/'metadata.json')
audit = load_json(REPORTS/'data_audit.json')
scores = table('metrics.csv')
test_scores = scores[(scores.split == 'test') & (scores.scope == 'all')].set_index('model')
q1, q2 = metadata['thresholds']
PAGES = ['Tổng quan dự án', 'Khám phá dữ liệu', 'Nhãn & mô hình',
         'Thực nghiệm & đánh giá', 'Dự đoán tương tác', 'Báo cáo & bàn giao']
with st.sidebar:
    st.title('🌫️ Air Quality Lab')
    st.caption('MODULE 02 · LOGISTIC REGRESSION')
    page = st.radio('Nội dung dự án', PAGES, key='page')
    st.divider()
    st.write('**Dự báo trước 1 giờ**')
    st.caption('12 trạm Bắc Kinh · 2013–2017\n\nGiờ dữ liệu: UTC+8')
    st.caption(f'Mô hình: {metadata["model_version"]}\n\nBộ đặc trưng: {metadata["feature_set"]}')
    st.info('Phân nhóm PM2.5 theo phân vị train. Không phải AQI chính thức.')
st.caption('ĐỀ TÀI DỰ ĐOÁN CHẤT LƯỢNG KHÔNG KHÍ / MODULE 2')
st.title(page)

if page == PAGES[0]:
    st.write('Từ các số đo đã có đến thời điểm **t**, Logistic Regression dự báo **nhóm nồng độ PM2.5 tại t+1 giờ** và xác suất của ba lớp. Khám phá dữ liệu, phương pháp, thực nghiệm và dự đoán qua các mục bên trái.')
    cols = st.columns(4)
    cols[0].metric('Bản ghi quan trắc', f'{audit["raw_rows"]:,}')
    cols[1].metric('Trạm quan trắc', len(audit['stations']))
    cols[2].metric('Accuracy · test', f'{test_scores.loc["logistic", "accuracy"]:.2%}')
    gain = test_scores.loc['logistic', 'macro_f1'] - test_scores.loc['persistence', 'macro_f1']
    cols[3].metric('Macro-F1 · test', f'{test_scores.loc["logistic", "macro_f1"]:.4f}', f'{gain:+.4f} so với persistence')
    st.subheader('Từ dữ liệu đến kết quả')
    for col, title, body in zip(st.columns(4), ['01 · Quan trắc', '02 · Đặc trưng', '03 · Học & chọn', '04 · Dự báo'], [
        'PM2.5, 5 chất ô nhiễm khác, thời tiết và thời gian tại 12 trạm.',
        'Lag đúng giờ, thống kê 3/6 giờ và chu kỳ giờ/tháng. Chỉ lấy thông tin đến t.',
        'Fit trên train, chọn bằng Macro-F1 validation, đánh giá cuối trên test.',
        'Thấp / Trung bình / Cao, kèm xác suất. Đối chiếu số đo thật giờ tiếp theo.',
    ]):
        with col.container(border=True):
            st.markdown(f'**{title}**')
            st.write(body)
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader('So sánh với baseline')
        st.bar_chart(test_scores[['accuracy', 'macro_f1']].rename(columns={'accuracy':'Accuracy','macro_f1':'Macro-F1'}), horizontal=True, stack=False,color=['#167d8d','#7299d2'])
        st.caption('Majority luôn chọn lớp phổ biến nhất trong tập huấn luyện. Persistence giữ nguyên nhóm PM2.5 tại giờ t. Cả ba mô hình được đánh giá trên cùng tập mẫu kiểm tra.')
    with right:
        st.subheader('Điểm cần trình bày khi bảo vệ')
        st.write(f'**Cải thiện nhỏ:** Macro-F1 tăng {gain:.4f} so với persistence. Mô hình không vượt baseline trên mọi tiêu chí.')
        st.write(f'**Recall Cao:** Logistic {test_scores.loc["logistic", "recall_high"]:.2%}, persistence {test_scores.loc["persistence", "recall_high"]:.2%}.')
        st.write(f'**Nhầm Cao thành Thấp:** Logistic {test_scores.loc["logistic", "high_to_low"]:.2%}, persistence {test_scores.loc["persistence", "high_to_low"]:.2%}.')
        st.caption('Dữ liệu lịch sử; chưa kiểm chứng tại Việt Nam, trạm mới hoặc vận hành thời gian thực.')
    st.subheader('Vị trí Module 2 trong đề tài')
    show_table(pd.DataFrame([
        ['Module 1', 'Linear Regression', 'Dự báo nồng độ PM2.5'],
        ['Module 2 · dashboard này', 'Logistic Regression', 'Dự báo nhóm PM2.5 và xác suất'],
        ['Module 3', 'KNN', 'Tìm thời điểm có điều kiện tương tự'],
        ['Module 4', 'Random Forest', 'Dự báo phi tuyến và phân tích đặc trưng'],
    ], columns=['Module','Thuật toán','Vai trò']), hide_index=True, width='stretch')
    st.caption('Bảng mô tả kiến trúc chung. Dashboard này hiển thị kết quả thực nghiệm của Module 2.')

elif page == PAGES[1]:
    frame = get_history()
    st.write('Khảo sát các số đo đã được kiểm tra tính hợp lệ. Mặc định hiển thị giai đoạn huấn luyện (Train); chọn giai đoạn khác để khám phá dữ liệu tương ứng.')
    a,b,c = st.columns([2,1,1])
    stations = a.multiselect('Trạm', metadata['stations'], default=['Aotizhongxin'], key='eda_stations')
    phase = b.selectbox('Giai đoạn', ['Train','Validation','Test','Toàn bộ'])
    variable = c.selectbox('Biến quan trắc', NUMERIC)
    subset = frame.loc[frame.station.isin(stations)].copy()
    if phase != 'Toàn bộ':
        start,end = metadata['splits'][phase.lower()]
        subset = subset.loc[(subset.timestamp >= start) & (subset.timestamp < end)]
    st.caption('Bộ lọc này dùng thời gian quan trắc t. Các tập huấn luyện/đánh giá chia theo thời gian mục tiêu t+1.')
    if subset.empty:
        st.info('Chọn ít nhất một trạm để xem biểu đồ.')
        st.stop()
    unit = metadata['schema'][variable]['unit']
    cols = st.columns(4)
    cols[0].metric('Số hàng đang xem', f'{len(subset):,}')
    cols[1].metric('Tỷ lệ thiếu biến đã chọn', f'{subset[variable].isna().mean():.2%}')
    cols[2].metric(f'Trung vị ({unit})', f'{subset[variable].median():.2f}')
    cols[3].metric(f'Phân vị 95% ({unit})', f'{subset[variable].quantile(.95):.2f}')
    st.subheader(f'Diễn biến {variable} · {unit}')
    daily = subset.groupby(['station', pd.Grouper(key='timestamp',freq='D')])[variable].mean().unstack('station')
    st.line_chart(daily)
    st.caption('Trung bình ngày từ số đo quan sát được; ngày không có số đo giữ thiếu. Không dùng biểu đồ này để điền mục tiêu.')
    left,right = st.columns(2)
    with left:
        st.subheader('Phân phối giá trị')
        values = subset[variable].dropna()
        if len(values):
            counts,edges = np.histogram(values,bins=35)
            st.bar_chart(pd.DataFrame({'Giá trị':(edges[:-1]+edges[1:])/2,'Số quan sát':counts}),x='Giá trị',y='Số quan sát')
    with right:
        st.subheader('Chu kỳ trung bình theo giờ')
        hourly = subset.groupby(subset.timestamp.dt.hour)[variable].mean()
        hourly.index.name = 'Giờ Bắc Kinh'
        st.line_chart(hourly)
    st.subheader('Dữ liệu thiếu theo biến trong lựa chọn')
    st.bar_chart((subset[NUMERIC+['wd']].isna().mean()*100).sort_values(ascending=False).rename('Thiếu (%)'),horizontal=True)
    with st.expander('Kiểm tra chất lượng toàn bộ dữ liệu'):
        st.write(f'Bản trùng hoàn toàn đã bỏ: **{audit["exact_duplicates_removed"]}** · Giờ bổ sung: **{audit["inserted_hours"]}**.')
        show_table(pd.Series(audit['invalid_values'],name='Giá trị không hợp lệ').to_frame(),width='stretch')
        show_table(table('missing_by_station.csv'),hide_index=True,width='stretch')
        st.caption('Nồng độ âm, giá trị vô hạn và áp suất không dương chuyển thành thiếu. Nhiệt độ âm vẫn hợp lệ. Không loại nồng độ cao chỉ vì là ngoại lệ.')
    with st.expander('Xem số đo và tải dữ liệu đang chọn'):
        st.caption('Bảng và tệp tải đều chứa toàn bộ dữ liệu phù hợp với bộ lọc. Có thể dùng nút phóng to ở góc trên bên phải bảng để xem rộng hơn.')
        show_table(subset,hide_index=True,width='stretch')
        download_csv(subset,'observations_filtered.csv')

elif page == PAGES[2]:
    st.subheader('Ba lớp được định nghĩa như thế nào?')
    for col,name,rule in zip(st.columns(3),CLASS_NAMES,[f'0 ≤ PM2.5 ≤ {q1:g}',f'{q1:g} < PM2.5 ≤ {q2:g}',f'PM2.5 > {q2:g}']):
        with col.container(border=True):
            st.markdown(f'### {name}')
            st.write(f'{rule} µg/m³')
    st.info('Ngưỡng là phân vị 1/3 và 2/3 của PM2.5 mục tiêu train, gộp 12 trạm. Thấp không đồng nghĩa không khí an toàn. Ngưỡng giữ nguyên khi đánh giá và demo.')
    counts = table('split_support.csv')
    support = counts.groupby(['split','label'])['count'].sum().unstack().rename(columns=dict(enumerate(CLASS_NAMES))).reindex(['train','validation','test'])
    st.subheader('Chia tập theo thời gian mục tiêu')
    show_table(pd.DataFrame([{'Tập':name,'Từ (gồm)':bounds[0],'Đến (không gồm)':bounds[1],'Số mẫu':int(support.loc[name].sum())} for name,bounds in metadata['splits'].items()]),hide_index=True,width='stretch')
    st.bar_chart(support,stack=True,color=['#319a91','#d9a441','#bd6578'])
    st.caption('Dữ liệu được chia theo thời gian, không chia ngẫu nhiên. Mẫu thiếu nhãn bị loại, không nội suy hoặc điền bằng trung vị. Lịch sử cuối tập huấn luyện có thể dùng ở đầu tập validation vì đã có tại thời điểm dự báo.')
    left,right = st.columns(2)
    with left:
        st.subheader('Pipeline xử lý')
        st.markdown('1. **Lưới giờ riêng từng trạm** → lag và thống kê quá khứ.\n2. **Numeric** → median train + cờ thiếu → StandardScaler.\n3. **station, wd** → giá trị missing → one-hot.\n4. **Logistic Regression** → softmax → nhãn và xác suất.')
        st.write(f'Cấu hình: **C={metadata["selected_run"]["C"]:g}**, class_weight=**{metadata["selected_run"]["class_weight"]}**, solver=**lbfgs**.')
    with right:
        st.subheader('Cơ sở mô hình')
        st.latex(r'P(y=k\mid x)=\frac{e^{w_k^Tx+b_k}}{\sum_j e^{w_j^Tx+b_j}}')
        st.write('Lớp có xác suất lớn nhất là dự đoán. C nhỏ tương ứng điều chuẩn L2 mạnh hơn. Chuẩn hóa giúp xử lý nhất quán các biến khác đơn vị.')
        st.caption('Mô hình không trực tiếp mô hình hóa thứ tự Thấp–Trung bình–Cao. Xác suất chưa được hiệu chỉnh trên dữ liệu độc lập.')
    with st.expander('Danh sách đầy đủ đặc trưng và đơn vị'):
        show_table(pd.DataFrame([{'Đặc trưng':k,'Kiểu':v['dtype'],'Đơn vị':v['unit']} for k,v in metadata['schema'].items()]),hide_index=True,width='stretch')
        st.caption('Lag t−1, −2, −3, −6, −12, −24 giờ. Rolling 3/6 giờ kết thúc tại t, min_periods=1, std ddof=1, kèm số quan sát. DEWP là điểm sương, không phải độ ẩm.')
    st.subheader('Hệ số theo lớp')
    class_name = st.selectbox('Lớp cần phân tích hệ số',CLASS_NAMES)
    coefficients = table('coefficients.csv').set_index('Unnamed: 0').rename_axis('Đặc trưng')
    group = st.radio('Nhóm hệ số',['Biến số đã chuẩn hóa','Biến phân loại one-hot'],horizontal=True,key='coefficient_group')
    prefix = 'numeric__' if group.startswith('Biến số') else 'categorical__'
    series = coefficients.loc[coefficients.index.str.startswith(prefix),class_name]
    top = series.loc[series.abs().nlargest(12).index].sort_values()
    st.bar_chart(top.rename('Hệ số'),horizontal=True)
    st.caption('Hệ số dương làm tăng điểm logit của lớp; hệ số âm làm giảm khi giữ các biến khác cố định. Không so trực tiếp độ lớn hệ số biến số với biến phân loại one-hot. Các biến trễ có tương quan; hệ số không thể hiện quan hệ nhân quả.')

elif page == PAGES[3]:
    split = st.selectbox('Tập đánh giá',['test','validation'],format_func=lambda x:'Test · đánh giá cuối' if x=='test' else 'Validation · chọn mô hình')
    scope = st.radio('Mẫu dùng để so sánh',['all','observed_current'],format_func=lambda x:'Tất cả mẫu có mục tiêu' if x=='all' else 'Chỉ mẫu có PM2.5(t) quan sát được',horizontal=True,key='evaluation_scope')
    selected = scores[(scores.split==split)&(scores.scope==scope)].set_index('model')
    cols = st.columns(4)
    cols[0].metric('Số mẫu mỗi mô hình',f'{int(selected.loc["logistic","n"]):,}')
    cols[1].metric('Accuracy · Logistic',f'{selected.loc["logistic","accuracy"]:.2%}')
    cols[2].metric('Macro-F1 · Logistic',f'{selected.loc["logistic","macro_f1"]:.4f}')
    cols[3].metric('Recall Cao · Logistic',f'{selected.loc["logistic","recall_high"]:.2%}')
    st.subheader('Logistic và hai baseline trên cùng mẫu')
    st.bar_chart(selected[['accuracy','macro_f1','recall_high']],stack=False,color=['#167d8d','#7299d2','#d9a441'])
    show_table(selected[['n','accuracy','macro_f1','recall_high','high_to_low']],width='stretch')
    st.caption(f'Persistence dùng majority khi PM2.5(t) thiếu: {selected.fallback_rate.iloc[0]:.2%} trên toàn tập {split}. Tập con quan sát được không cần fallback. Macro-F1 lấy trung bình F1 của cả ba lớp; Cao → Thấp tính trên số mẫu thật thuộc lớp Cao.')
    st.subheader('Confusion matrix và chất lượng từng lớp')
    st.caption(f'Hình và báo cáo từng lớp bên dưới luôn dùng TOÀN BỘ mẫu {split}, độc lập với bộ lọc tập con phía trên.')
    for col,kind,title in zip(st.columns(2),['count','true'],['Số mẫu','Tỷ lệ theo lớp thật']):
        path=REPORTS/f'confusion_{split}_{kind}.png'
        if path.exists():col.image(str(path),caption=title,width='stretch')
    details = load_json(REPORTS/'classification_report.json')
    show_table(pd.DataFrame(details[split]).T.loc[CLASS_NAMES],width='stretch')
    st.caption('Hàng là lớp thật, cột là dự đoán. Low = Thấp, Medium = Trung bình, High = Cao.')
    with st.expander('Nhật ký chọn mô hình trên validation',expanded=True):
        experiments=table('experiments.csv').copy()
        experiments['Cấu hình']=experiments.apply(lambda r:f'{r.feature_set} · C={r.C:g} · {r.class_weight if pd.notna(r.class_weight) else "None"}',axis=1)
        st.bar_chart(experiments.set_index('Cấu hình')['macro_f1'],horizontal=True)
        show_table(experiments,hide_index=True,width='stretch')
        st.caption('So sánh 3 bộ đặc trưng ở C=1 rồi thử C/weight trên bộ đã chọn. Mô hình cuối giữ fit trên train; không chọn lại theo test.')
    st.subheader('Lỗi theo trạm, tháng và giờ trên test')
    dimension = st.selectbox('Chiều phân tích',['station','month','hour'],format_func=lambda x:{'station':'Trạm','month':'Tháng','hour':'Giờ'}[x])
    slices=table('test_slices.csv')
    slices=slices.loc[slices.dimension==dimension].copy()
    if dimension!='station':slices=slices.assign(order=pd.to_numeric(slices.value)).sort_values('order').drop(columns='order')
    st.bar_chart(slices.set_index('value')[['macro_f1','recall_high']],stack=False,color=['#167d8d','#7299d2'])
    show_table(slices.drop(columns='dimension'),hide_index=True,width='stretch')
    st.write(f'**Giờ chuyển lớp:** {details["transition_support"]:,} mẫu test có nhóm t+1 khác nhóm t và quan sát được PM2.5(t).')
    show_table(pd.DataFrame(details['transition_metrics']).T,width='stretch')
    st.caption('Phân tích lát cắt chỉ mô tả kết quả; không dùng để chỉnh mô hình sau khi xem test.')
    with st.expander('Tra cứu dự đoán test'):
        predictions=table('test_predictions.csv')
        station=st.selectbox('Trạm trong bảng test',metadata['stations'])
        error_only=st.checkbox('Chỉ xem dự đoán sai')
        filtered=predictions.loc[predictions.station==station]
        if error_only:filtered=filtered.loc[filtered.label!=filtered.logistic]
        st.caption(f'{len(filtered):,} mẫu phù hợp với bộ lọc. Mã lớp 0 = Thấp, 1 = Trung bình, 2 = Cao. Cột label là lớp thật; logistic là lớp dự đoán.')
        show_table(filtered,hide_index=True,width='stretch')
        download_csv(filtered,'test_predictions_filtered.csv')

elif page == PAGES[4]:
    frame=get_history()
    st.write('Chọn thời điểm trong giai đoạn test. Đầu vào chỉ lấy lịch sử đến t; số đo t+1 được mở để đối chiếu sau dự đoán.')
    a,b,c=st.columns([2,2,1])
    station=a.selectbox('Trạm dự đoán',metadata['stations'],key='prediction_station')
    available=frame.loc[(frame.station==station)&(frame.timestamp>pd.Timestamp(metadata['training_target_end']))&
        (frame.timestamp+pd.Timedelta(hours=1)>=metadata['splits']['test'][0])&
        (frame.timestamp+pd.Timedelta(hours=1)<metadata['splits']['test'][1])]
    date=b.date_input('Ngày · giờ Bắc Kinh UTC+8',value=pd.Timestamp('2016-05-01').date(),min_value=available.timestamp.min().date(),max_value=available.timestamp.max().date())
    hours=available.loc[available.timestamp.dt.date==date,'timestamp'].dt.hour.tolist()
    hour=c.selectbox('Giờ',hours,index=hours.index(9) if 9 in hours else 0)
    time=pd.Timestamp(date)+pd.Timedelta(hours=hour)
    features=build_features(frame,station,time)
    st.caption(f'Dữ liệu đến {time:%d/%m/%Y %H:%M} → dự báo {(time+pd.Timedelta(hours=1)):%d/%m/%Y %H:%M} · UTC+8')
    if st.button('Dự đoán giờ tiếp theo',type='primary',key='predict'):
        try:
            result=predict_pollution_class(features)
            st.session_state['prediction']=(station,time,result)
        except (ValueError,FileNotFoundError) as exc:
            st.error(str(exc))
    saved=st.session_state.get('prediction')
    if saved and saved[:2]==(station,time):
        result=saved[2]
        st.subheader('Kết quả dự báo')
        a,b,c=st.columns(3)
        a.metric('Nhóm dự đoán',result['predicted_class'])
        b.metric('Xác suất lớp dự đoán',f'{max(result["probabilities"].values()):.2%}')
        c.metric('Đặc trưng thiếu',result['missing_feature_count'])
        st.bar_chart(pd.Series(result['probabilities'],name='Xác suất'),horizontal=True,color='#167d8d')
        st.caption('Xác suất chưa được hiệu chỉnh; không phải mức chắc chắn đã được kiểm chứng hoặc chỉ số sức khỏe.')
        if result['missing_feature_count']:st.warning('Pipeline xử lý đặc trưng thiếu bằng quy tắc train. Xem bảng đầu vào để biết giá trị nào thiếu.')
        with st.expander('Đối chiếu thực tế t+1 và hai baseline'):
            actual=frame.loc[(frame.station==station)&(frame.timestamp==time+pd.Timedelta(hours=1)),'PM2.5']
            majority=CLASS_NAMES[metadata['majority_class']]
            persistence=CLASS_NAMES[int(labels([features['PM2.5']],metadata['thresholds'])[0])] if pd.notna(features['PM2.5']) else majority
            if actual.empty or pd.isna(actual.iloc[0]):st.info('Không có số đo mục tiêu hợp lệ để xác định đúng/sai.')
            else:
                actual_class=CLASS_NAMES[int(labels([actual.iloc[0]],metadata['thresholds'])[0])]
                st.write(f'**PM2.5 thực tế: {actual.iloc[0]:g} µg/m³ · {actual_class}**')
                comparison=pd.DataFrame({'Phương pháp':['Logistic','Persistence','Majority'],'Dự đoán':[result['predicted_class'],persistence,majority]})
                comparison['Đối chiếu']=np.where(comparison['Dự đoán']==actual_class,'Đúng','Sai')
                show_table(comparison,hide_index=True,width='stretch')
            if pd.isna(features['PM2.5']):st.caption('Persistence dùng fallback majority vì PM2.5(t) thiếu.')
        st.download_button('Tải dự đoán JSON',json.dumps(result,ensure_ascii=False,indent=2),'prediction.json','application/json')
    else:st.info('Nhấn “Dự đoán giờ tiếp theo” để tính kết quả cho lựa chọn hiện tại.')
    st.subheader('Điều kiện tại thời điểm t')
    for col,key in zip(st.columns(4),['PM2.5','TEMP','PRES','WSPM']):
        value=features[key]
        col.metric(f'{key} · {metadata["schema"][key]["unit"]}',f'{value:g}' if pd.notna(value) else 'Thiếu')
    past=frame.loc[(frame.station==station)&(frame.timestamp<=time)&(frame.timestamp>=time-pd.Timedelta(hours=48))]
    st.subheader('48 giờ quan trắc gần nhất')
    trend=past.set_index('timestamp')[['PM2.5']].copy()
    trend['Ngưỡng Thấp / Trung bình']=q1
    trend['Ngưỡng Trung bình / Cao']=q2
    st.line_chart(trend,color=['#167d8d','#d9a441','#bd6578'])
    st.caption('Đường ngưỡng là ranh giới phân vị train. Chuỗi dừng tại t, không chứa số đo tương lai.')
    with st.expander('Toàn bộ đặc trưng gửi vào mô hình'):
        show_table(pd.DataFrame([{'Đặc trưng':k,'Giá trị':str(features[k]) if pd.notna(features[k]) else 'Thiếu → pipeline xử lý','Đơn vị':metadata['schema'][k]['unit']} for k in metadata['features']]),hide_index=True,width='stretch')
        st.caption('Các bước xử lý được tải từ mô hình đã lưu, không học lại khi dự đoán. Giờ và tháng được biểu diễn bằng sin/cos; day_of_week mã hóa thứ Hai = 0 đến Chủ nhật = 6.')

else:
    st.write('Tài liệu, kết quả và hướng dẫn tái lập phục vụ thuyết trình và tích hợp Module 2.')
    if (REPORTS/'REPORT.md').exists():st.markdown((REPORTS/'REPORT.md').read_text(encoding='utf-8'))
    st.subheader('Tải kết quả dự án')
    for col,(path,title,mime) in zip(st.columns(3),[
        (ROOT/'module2.ipynb','Notebook đã thực thi','application/x-ipynb+json'),
        (REPORTS/'REPORT.md','Báo cáo Markdown','text/markdown'),
        (REPORTS/'metrics.csv','Bảng đánh giá CSV','text/csv'),
    ]):
        if path.exists():col.download_button(title,path.read_bytes(),path.name,mime)
    with st.expander('Schema và phiên bản môi trường'):
        st.json(metadata)
        st.download_button('Tải metadata JSON',json.dumps(metadata,ensure_ascii=False,indent=2),'metadata.json','application/json')
    st.subheader('Chạy lại và kiểm tra')
    st.code('cd "Module 2"\npython -m pip install -r requirements.txt\npython train.py\npython -m unittest -v test_module2.py test_app.py\npython -m streamlit run app.py --server.port 8503',language='powershell')
    st.caption('Chỉ chạy cd khi terminal ở thư mục ML. Huấn luyện lại ghi đè kết quả sinh tự động; giữ nguyên raw.')
    st.subheader('Tích hợp vào ứng dụng chung')
    st.code("from module2 import load_data, build_features, predict_pollution_class\nhistory, _, _ = load_data()\nfeatures = build_features(history, 'Aotizhongxin', '2016-05-01 09:00')\nresult = predict_pollution_class(features)",language='python')
    st.write('Nguồn: Chen, S. (2017), UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G, CC BY 4.0. Không xác định thời điểm tải gốc; checksum CSV được lưu trong data_audit.json.')

st.divider()
st.caption('Air Quality Lab · Module 2 | Dữ liệu lịch sử Bắc Kinh | Phân nhóm tương đối, không phải AQI chính thức')
