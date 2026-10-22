"""Run feature selection, tuning and a single final test evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import time
import warnings
import platform
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from module2 import *

def metrics(y, prediction):
    matrix = confusion_matrix(y, prediction, labels=[0,1,2])
    high = int(matrix[2].sum())
    return dict(n=len(y), accuracy=float(accuracy_score(y,prediction)),
                macro_f1=float(f1_score(y,prediction,labels=[0,1,2],average='macro',zero_division=0)),
                recall_high=float(matrix[2,2]/high) if high else 0.,
                high_to_low=float(matrix[2,0]/high) if high else 0.)

def main():
    for directory in [ARTIFACTS, REPORTS]:
        directory.mkdir(exist_ok=True)
    frame, audit, missing = load_data()
    write_json(REPORTS/'data_audit.json',audit)
    missing.to_csv(REPORTS/'missing_by_station.csv')
    table = supervised_table(frame)
    masks = split_masks(table)
    train, val, test = [table.loc[masks[name]].copy() for name in ('train','validation','test')]
    thresholds = train.pm25_target.quantile([1/3,2/3]).to_numpy()
    for subset in [train,val,test]:
        subset['label'] = labels(subset.pm25_target,thresholds)
    if set(train.label) != {0,1,2}:
        raise ValueError('Train thiếu lớp.')
    majority = int(train.label.mode().iloc[0])
    policy = dict(strategy='train_quantiles_v1', thresholds=thresholds.tolist(), classes=CLASS_NAMES,
                  boundaries=['0 <= p <= q1','q1 < p <= q2','p > q2'], unit='µg/m³',
                  fit_target_time=['2013-03-01','2015-03-01 (exclusive)'], official_aqi=False)
    write_json(ARTIFACTS/'label_policy.json',policy)
    counts = pd.concat([x.groupby(['station','label']).size().rename('count').reset_index().assign(split=name)
                        for name,x in [('train',train),('validation',val),('test',test)]])
    counts.to_csv(REPORTS/'split_support.csv',index=False)
    fig, axes = plt.subplots(2,1,figsize=(11,7))
    train['PM2.5'].hist(bins=80,ax=axes[0]); axes[0].set(title='Train PM2.5 distribution',xlabel='PM2.5 (µg/m³)')
    train.loc[train.station=='Aotizhongxin'].set_index('timestamp')['PM2.5'].resample('D').mean().plot(ax=axes[1])
    axes[1].set(title='Aotizhongxin train daily mean',ylabel='PM2.5 (µg/m³)')
    fig.tight_layout(); fig.savefig(REPORTS/'eda.png'); plt.close(fig)
    runs=[]
    def fit(feature_set,C=1.,weight=None):
        columns=FEATURE_SETS[feature_set]+CATEGORIES
        if train[FEATURE_SETS[feature_set]].isna().all().any():
            raise ValueError('Có cột số hoàn toàn thiếu trong train.')
        model=make_pipeline(feature_set,C,weight)
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter('always')
            model.fit(train[columns],train.label)
        warning_text=[str(w.message) for w in caught]
        score=metrics(val.label,model.predict(val[columns]))
        row=dict(feature_set=feature_set,C=C,class_weight=weight,fit_seconds=time.perf_counter()-start,
                 warnings=warning_text,converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),**score)
        runs.append(row)
        print(f'{feature_set} C={C} weight={weight}: val F1={score["macro_f1"]:.5f}, {row["fit_seconds"]:.1f}s',flush=True)
        pd.DataFrame(runs).to_csv(REPORTS/'experiments.csv',index=False)
        if not row['converged']:
            raise RuntimeError('Chưa hội tụ; xem experiments.csv trước khi tăng max_iter.')
        return model,row
    candidates=[fit(name) for name in FEATURE_SETS]
    best_model,best=max(candidates,key=lambda item:item[1]['macro_f1'])
    selected=best['feature_set']
    for C in (.01,.1,1.,10.):
        for weight in (None,'balanced'):
            if C == 1 and weight is None:
                continue
            model,result=fit(selected,C,weight)
            if result['macro_f1'] > best['macro_f1'] + 1e-5:
                best_model,best=model,result
    columns=FEATURE_SETS[selected]+CATEGORIES
    # Keep the selected train-only fit: no refit is necessary, thresholds stay frozen.
    joblib.dump(best_model,ARTIFACTS/'model.joblib')
    np.testing.assert_allclose(best_model.predict_proba(val[columns].iloc[:20]),joblib.load(ARTIFACTS/'model.joblib').predict_proba(val[columns].iloc[:20]))
    scores=[]
    details={}
    for name,subset in [('validation',val),('test',test)]:
        observed=subset['PM2.5'].notna().to_numpy()
        persistence=np.full(len(subset),majority)
        persistence[observed]=labels(subset.loc[observed,'PM2.5'],thresholds)
        predictions={'majority':np.full(len(subset),majority),'persistence':persistence,'logistic':best_model.predict(subset[columns])}
        for model_name,pred in predictions.items():
            for scope,mask in [('all',np.ones(len(subset),dtype=bool)),('observed_current',observed)]:
                scores.append(dict(split=name,model=model_name,scope=scope, fallback_rate=float((~observed).mean()),**metrics(subset.label.to_numpy()[mask],pred[mask])))
        pred=predictions['logistic']
        details[name]=classification_report(subset.label,pred,labels=[0,1,2],target_names=CLASS_NAMES,zero_division=0,output_dict=True)
        for normalize in (None,'true'):
            fig,ax=plt.subplots(figsize=(6,5))
            ConfusionMatrixDisplay.from_predictions(subset.label,pred,labels=[0,1,2],display_labels=['Low','Medium','High'],normalize=normalize,values_format='d' if normalize is None else '.3f',ax=ax,colorbar=False)
            ax.set_title(f'{name} confusion matrix'); fig.tight_layout()
            fig.savefig(REPORTS/f'confusion_{name}_{normalize or "count"}.png');plt.close(fig)
        if name=='test':
            output=subset[['station','timestamp','target_time','PM2.5','pm25_target','label']].copy()
            for model_name,prediction in predictions.items(): output[model_name]=prediction
            for k,p in zip(best_model.classes_,best_model.predict_proba(subset[columns]).T):output[f'probability_{k}']=p
            output.to_csv(REPORTS/'test_predictions.csv',index=False)
            slices=[]
            for key,values in [('station',subset.station),('month',subset.target_time.dt.month),('hour',subset.target_time.dt.hour)]:
                for value in sorted(values.unique()):
                    mask=(values==value).to_numpy()
                    slices.append(dict(dimension=key,value=value,**metrics(subset.label.to_numpy()[mask],pred[mask])))
            pd.DataFrame(slices).to_csv(REPORTS/'test_slices.csv',index=False)
            transition=observed & (persistence != subset.label.to_numpy())
            details['transition_support']=int(transition.sum())
            details['transition_metrics']={k:metrics(subset.label.to_numpy()[transition],p[transition]) for k,p in predictions.items()}
    pd.DataFrame(scores).to_csv(REPORTS/'metrics.csv',index=False)
    write_json(REPORTS/'classification_report.json',details)
    names=best_model['preprocess'].get_feature_names_out()
    coefficients=pd.DataFrame(best_model['model'].coef_.T,index=names,columns=CLASS_NAMES)
    coefficients.to_csv(REPORTS/'coefficients.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for label,ax in zip(CLASS_NAMES,axes):
        s=coefficients[label];s.loc[s.abs().nlargest(10).index].sort_values().plot.barh(ax=ax);ax.set_title(label)
    fig.tight_layout();fig.savefig(REPORTS/'coefficients.png');plt.close(fig)
    metadata=dict(model_version='module2_v1',feature_set=selected,features=columns,stations=audit['stations'],
                  training_target_end=str(train.target_time.max()),training_as_of_end=str(train.timestamp.max()),
                  splits={'train':['2013-03-01','2015-03-01'],'validation':['2015-03-01','2016-03-01'],'test':['2016-03-01','2017-03-01']},
                  selected_run=best,refit_train_validation=False,majority_class=majority,thresholds=thresholds.tolist(),
                  versions={p:version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','joblib','streamlit','nbformat','nbclient']},
                  python=platform.python_version(),created_at=datetime.now(timezone.utc).isoformat(),sha256=audit['sha256'],
                  code_sha256={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [ROOT/'module2.py',ROOT/'train.py']},
                  timezone='Asia/Shanghai; naive inputs interpreted as Beijing time',
                  schema={c:dict(dtype='string' if c in CATEGORIES else 'float',unit='µg/m³' if c in NUMERIC[:6] or c.startswith('pm25_lag') or c in ['mean_3','mean_6','std_3','std_6','trend'] else {'TEMP':'°C','DEWP':'°C','PRES':'hPa','RAIN':'mm','WSPM':'m/s'}.get(c,'dimensionless')) for c in columns},
                  preprocessing='Train-only median + missing indicators + StandardScaler; missing category + one-hot',
                  rolling='Trailing 3/6 hours including t; min_periods=1; std ddof=1; counts included',metrics=scores)
    write_json(ARTIFACTS/'metadata.json',metadata)
    example=build_features(frame,'Aotizhongxin','2016-05-01 09:00')
    write_json(REPORTS/'prediction_example.json',predict_pollution_class(example))
    result=pd.DataFrame(scores)
    rows=result[(result.split=='test') & (result.scope=='all')]
    report=['# Báo cáo Module 2 Logistic Regression','',
            'Dự báo nhóm nồng độ PM2.5 tại t+1 giờ, dùng số đo đã có đến t. Đây là dự báo cuốn chiếu; không phải AQI chính thức hoặc khuyến cáo sức khỏe.','',
            f'Dữ liệu: {audit["raw_rows"]:,} dòng, {len(audit["stations"])} trạm. Số mẫu train/validation/test: {len(train):,} / {len(val):,} / {len(test):,}.',
            f'Ngưỡng train: q1={thresholds[0]:.12g}, q2={thresholds[1]:.12g} µg/m³. Thấp: p≤q1; Trung bình: q1<p≤q2; Cao: p>q2.',
            f'Cấu hình chọn bằng validation: {selected}, C={best["C"]}, class_weight={best["class_weight"]}. Không refit train+validation.','',
            '| Mô hình | Accuracy | Macro-F1 | Recall Cao | Cao → Thấp |','|---|---:|---:|---:|---:|']
    for row in rows.itertuples():report.append(f'| {row.model} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.recall_high:.4f} | {row.high_to_low:.4f} |')
    logistic=rows.loc[rows.model=='logistic','macro_f1'].iloc[0]; persistence=rows.loc[rows.model=='persistence','macro_f1'].iloc[0]
    report += ['',f'Chênh lệch Macro-F1 Logistic so với persistence: {logistic-persistence:+.4f}. Persistence mạnh vì nồng độ thường ổn định giữa hai giờ liền nhau.',
               f'Tỷ lệ fallback persistence trên test: {rows.fallback_rate.iloc[0]:.2%}. So sánh trên cùng tập con có PM2.5(t) nằm trong metrics.csv.',
               'Cần đọc Recall Cao cùng Macro-F1: cải thiện điểm trung bình không đảm bảo nhận diện lớp Cao tốt hơn. Confusion matrix thể hiện cả lỗi Cao → Thấp và Cao → Trung bình.',
               f'Có {details["transition_support"]:,} giờ chuyển lớp khi PM2.5(t) quan sát được; Macro-F1 Logistic tại các giờ này: {details["transition_metrics"]["logistic"]["macro_f1"]:.4f}.',
               '', '## Phương pháp và diễn giải',
               'Logistic Regression đa lớp dùng softmax P(y=k|x)=exp(w_k·x+b_k)/Σ_j exp(w_j·x+b_j). C nhỏ tăng điều chuẩn L2. Numeric được điền median và chuẩn hóa chỉ từ train; station và wd dùng one-hot. Macro-F1 là trung bình F1 của ba lớp, kể cả khi một lát cắt thiếu lớp.',
               'Lag theo từng trạm trên lưới giờ; không nội suy mục tiêu. Chia theo target_time để nhãn train không tràn sang validation. Thử ba bộ đặc trưng ở C=1 rồi tối đa tám cấu hình C/weight trên bộ được chọn; chỉ đánh giá test sau khi khóa cấu hình.',
               'Hệ số dương làm tăng logit của lớp tương ứng khi giữ các biến khác cố định. Hệ số numeric gắn với một độ lệch chuẩn; không so trực tiếp với one-hot và không suy diễn nhân quả. PM2.5 và các lag tương quan nên hệ số có thể khó diễn giải.',
               '', '## Hạn chế',
               'Nhãn phân vị là mức tương đối tại Bắc Kinh, không phải mức an toàn sức khỏe. Mẫu thiếu mục tiêu bị loại có thể gây thiên lệch. Xác suất chưa được hiệu chỉnh. Chưa đánh giá trạm mới, Việt Nam hoặc dữ liệu thời gian thực. Mô hình tuyến tính có thể bỏ lỡ biến động đột ngột.',
               'Nguồn: Chen, S. (2017), UCI Beijing Multi-Site Air Quality, DOI 10.24432/C5RK5G, CC BY 4.0. CSV được cung cấp sẵn; không xác định được thời điểm tải gốc. SHA-256 được lưu trong data_audit.json.',
               '', '## Tệp kết quả',
               'experiments.csv: toàn bộ lượt huấn luyện và cảnh báo; classification_report.json: precision/recall/F1/support; test_slices.csv: trạm/tháng/giờ và support; test_predictions.csv: nhãn, baseline và xác suất từng mẫu; coefficients.csv: hệ số theo lớp.']
    report += ['', '## Phân tích theo trạm và thời gian', 'Các lát cắt dưới đây chỉ dùng để phân tích sau đánh giá, không dùng để chọn lại mô hình.']
    for dimension,group in pd.DataFrame(slices).groupby('dimension'):
        worst=group.loc[group.macro_f1.idxmin()]
        report.append(f'Lát cắt {dimension} thấp nhất: {worst["value"]}, Macro-F1={worst.macro_f1:.4f}, support={worst.n}.')
    (REPORTS/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in metadata['versions'].items())+'\n',encoding='utf-8')
    print(rows.to_string(index=False),flush=True)
    return metadata

if __name__=='__main__':main()
