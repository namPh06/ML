"""Shared causal features and prediction API for Module 2."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'data'
ARTIFACTS = ROOT / 'artifacts'
REPORTS = ROOT / 'reports'
NUMERIC = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
NONNEGATIVE = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']
CATEGORIES = ['wd', 'station']
CYCLE = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week']
LAGS = [f'pm25_lag_{i}' for i in (1, 2, 3)]
EXTRA = [f'pm25_lag_{i}' for i in (6, 12, 24)] + ['mean_3', 'std_3', 'count_3', 'mean_6', 'std_6', 'count_6', 'trend']
FEATURE_SETS = {'basic': NUMERIC + CYCLE, 'lag': NUMERIC + CYCLE + LAGS, 'rolling': NUMERIC + CYCLE + LAGS + EXTRA}
CLASS_NAMES = ['Thấp', 'Trung bình', 'Cao']

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def load_data(data_dir=DATA):
    files = sorted(Path(data_dir).glob('PRSA_Data_*.csv'))
    if not files:
        raise ValueError('Không tìm thấy CSV PRSA.')
    frames, hashes = [], {}
    required = NUMERIC + CATEGORIES + ['year', 'month', 'day', 'hour']
    for path in files:
        frame = pd.read_csv(path)
        if set(required) - set(frame):
            raise ValueError(f'{path.name}: thiếu cột {set(required) - set(frame)}')
        frame = frame[required].copy()
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors='raise')
        frame['timestamp'] = pd.to_datetime(frame[['year', 'month', 'day', 'hour']], errors='raise')
        if frame.station.isna().any():
            raise ValueError('Thiếu station.')
        frames.append(frame.drop(columns=['year', 'month', 'day', 'hour']))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = pd.concat(frames, ignore_index=True)
    original_rows = len(raw)
    raw = raw.drop_duplicates()
    if raw.duplicated(['station', 'timestamp']).any():
        raise ValueError('Bản ghi trùng station/timestamp có giá trị mâu thuẫn.')
    invalid = {}
    for col in NUMERIC:
        bad = raw[col].notna() & ~np.isfinite(raw[col])
        if col in NONNEGATIVE:
            bad |= raw[col] < 0
        if col == 'PRES':
            bad |= raw[col] <= 0
        invalid[col] = int(bad.sum())
        raw.loc[bad, col] = np.nan
    missing = raw.groupby('station')[NUMERIC + ['wd']].agg(lambda x: int(x.isna().sum()))
    groups = []
    for station, group in raw.groupby('station', sort=True):
        group = group.set_index('timestamp').sort_index()
        grid = pd.date_range(group.index.min(), group.index.max(), freq='h', name='timestamp')
        group = group.reindex(grid)
        group['inserted_hour'] = group.station.isna()
        group['station'] = station
        groups.append(group.reset_index())
    frame = pd.concat(groups, ignore_index=True)
    audit = dict(raw_rows=original_rows, exact_duplicates_removed=original_rows-len(raw),
                 rows_on_hourly_grid=len(frame), inserted_hours=int(frame.inserted_hour.sum()),
                 stations=sorted(frame.station.unique()), start=str(frame.timestamp.min()),
                 end=str(frame.timestamp.max()), invalid_values=invalid, sha256=hashes,
                 source='https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data',
                 attribution='Chen, S. (2017), DOI 10.24432/C5RK5G, CC BY 4.0',
                 provenance='Existing shared CSVs; original download time unknown.', timezone='Asia/Shanghai')
    return frame, audit, missing

def feature_table(frame):
    out = frame.sort_values(['station', 'timestamp']).copy()
    if out.duplicated(['station', 'timestamp']).any():
        raise ValueError('Duplicate station/time.')
    gaps = out.groupby('station').timestamp.diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Feature table requires a regular hourly grid.')
    grouped = out.groupby('station')['PM2.5']
    for lag in (1, 2, 3, 6, 12, 24):
        out[f'pm25_lag_{lag}'] = grouped.shift(lag)
    for window in (3, 6):
        for stat in ('mean', 'std', 'count'):
            out[f'{stat}_{window}'] = grouped.transform(lambda x: getattr(x.rolling(window, min_periods=1), stat)())
    out['trend'] = out['PM2.5'] - out.pm25_lag_1
    hour = out.timestamp.dt.hour
    month = out.timestamp.dt.month - 1
    for name, value, period in [('hour', hour, 24), ('month', month, 12)]:
        out[f'{name}_sin'] = np.sin(2*np.pi*value/period)
        out[f'{name}_cos'] = np.cos(2*np.pi*value/period)
    out['day_of_week'] = out.timestamp.dt.dayofweek
    return out

def supervised_table(frame):
    out = feature_table(frame)
    out['target_time'] = out.timestamp + pd.Timedelta(hours=1)
    out['pm25_target'] = out.groupby('station')['PM2.5'].shift(-1)
    return out.loc[out.pm25_target.notna()].copy()

def split_masks(frame):
    t = frame.target_time
    return {'train': (t >= '2013-03-01') & (t < '2015-03-01'),
            'validation': (t >= '2015-03-01') & (t < '2016-03-01'),
            'test': (t >= '2016-03-01') & (t < '2017-03-01')}

def labels(values, thresholds):
    values = np.asarray(values, dtype=float)
    if len(thresholds) != 2 or not 0 <= thresholds[0] < thresholds[1] or not np.isfinite(thresholds).all():
        raise ValueError('Ngưỡng không hợp lệ.')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Không tạo nhãn từ mục tiêu thiếu/không hợp lệ.')
    return np.searchsorted(thresholds, values, side='left')

def make_pipeline(feature_set='lag', C=1, class_weight=None):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocess = ColumnTransformer([('numeric', numeric, FEATURE_SETS[feature_set]), ('categorical', categorical, CATEGORIES)])
    return Pipeline([('preprocess', preprocess), ('model', LogisticRegression(C=C, class_weight=class_weight, solver='lbfgs', max_iter=2000))])

def local_time(value):
    time = pd.Timestamp(value)
    if pd.isna(time):
        raise ValueError('Thiếu thời điểm.')
    if time.tzinfo is not None:
        time = time.tz_convert('Asia/Shanghai').tz_localize(None)
    if time != time.floor('h'):
        raise ValueError('Thời điểm phải là giờ tròn.')
    return time

def build_features(history_df, station, as_of_time):
    time = local_time(as_of_time)
    required = NUMERIC + CATEGORIES + ['timestamp']
    if set(required) - set(history_df):
        raise ValueError(f'Thiếu cột: {set(required) - set(history_df)}')
    history = history_df.loc[(history_df.station == station) & (history_df.timestamp <= time), required].copy()
    if history.empty or time not in history.timestamp.values:
        raise ValueError('Không có dữ liệu trạm tại giờ yêu cầu.')
    history = history.loc[history.timestamp >= time - pd.Timedelta(hours=24)]
    history = history.set_index('timestamp').reindex(pd.date_range(time-pd.Timedelta(hours=24), time, freq='h', name='timestamp'))
    history['station'] = station
    return feature_table(history.reset_index()).iloc[-1].to_dict()

def predict_pollution_class(input_data, artifact_dir=ARTIFACTS):
    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / 'metadata.json').read_text(encoding='utf-8'))
    required = metadata['features'] + ['timestamp']
    if set(required) - set(input_data):
        raise ValueError(f'Thiếu cột: {set(required) - set(input_data)}')
    if input_data['station'] not in metadata['stations']:
        raise ValueError('Trạm chưa được huấn luyện/đánh giá.')
    time = local_time(input_data['timestamp'])
    row = pd.DataFrame([input_data])[metadata['features']]
    for col in FEATURE_SETS[metadata['feature_set']]:
        row[col] = pd.to_numeric(row[col], errors='raise')
        value = row[col].iloc[0]
        if pd.notna(value) and (not np.isfinite(value) or ((col in NONNEGATIVE or col.startswith('pm25_lag_')) and value < 0) or (col == 'PRES' and value <= 0)):
            raise ValueError(f'Giá trị không hợp lệ: {col}')
    # Only load trusted, locally trained joblib files.
    model = joblib.load(artifact_dir / 'model.joblib')
    probability = model.predict_proba(row)[0]
    classes = model.classes_
    return dict(station=input_data['station'], as_of_time=time.tz_localize('Asia/Shanghai').isoformat(),
                target_time=(time+pd.Timedelta(hours=1)).tz_localize('Asia/Shanghai').isoformat(),
                predicted_class=CLASS_NAMES[int(classes[np.argmax(probability)])],
                probabilities={CLASS_NAMES[int(k)]: float(p) for k,p in zip(classes, probability)},
                label_policy='train_quantiles_v1', model_version='module2_v1',
                missing_feature_count=int(row.isna().sum().sum()))
