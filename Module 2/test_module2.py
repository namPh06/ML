import json
import unittest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from module2 import *

class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame,_,_=load_data()

    def test_labels_boundaries_and_invalid(self):
        np.testing.assert_array_equal(labels([0,10,10.1,20,21],[10,20]),[0,0,1,1,2])
        for values,thresholds in [([np.nan],[10,20]),([-1],[10,20]),([1],[10,10])]:
            with self.assertRaises(ValueError):labels(values,thresholds)

    def test_targets_lags_and_split_boundaries(self):
        table=supervised_table(self.frame)
        source=self.frame.set_index(['station','timestamp'])['PM2.5']
        for station,time in [('Aotizhongxin','2013-03-01 00:00'),('Aotizhongxin','2014-01-01 23:00'),('Changping','2014-05-01 09:00'),('Dingling','2015-02-28 23:00'),('Wanliu','2016-02-29 23:00')]:
            time=pd.Timestamp(time)
            row=table.loc[(table.station==station)&(table.timestamp==time)].iloc[0]
            self.assertEqual(row.pm25_target,source.loc[(station,time+pd.Timedelta(hours=1))])
        engineered=feature_table(self.frame)
        self.assertTrue(engineered.groupby('station').head(1).pm25_lag_1.isna().all())
        masks=split_masks(table)
        self.assertTrue((sum(masks.values())==1).all())
        self.assertLess(table.loc[masks['train'],'target_time'].max(),table.loc[masks['validation'],'target_time'].min())
        self.assertTrue(table.pm25_target.notna().all())

    def test_gap_and_future_invariance(self):
        station='Aotizhongxin';time=pd.Timestamp('2016-05-01 09:00')
        original=build_features(self.frame,station,time)
        altered=self.frame.copy()
        altered.loc[altered.timestamp>time,NUMERIC]=99999
        changed=build_features(altered,station,time)
        pd.testing.assert_series_equal(pd.Series(original),pd.Series(changed))
        gap=self.frame.loc[~((self.frame.station==station)&(self.frame.timestamp==time-pd.Timedelta(hours=1)))]
        self.assertTrue(pd.isna(build_features(gap,station,time)['pm25_lag_1']))
        batch=feature_table(self.frame)
        row=batch.loc[(batch.station==station)&(batch.timestamp==time)].iloc[0]
        np.testing.assert_allclose(row[FEATURE_SETS['rolling']].astype(float),pd.Series(original)[FEATURE_SETS['rolling']].astype(float),equal_nan=True)

    def test_artifact_and_preprocessing_contract(self):
        metadata=json.loads((ARTIFACTS/'metadata.json').read_text(encoding='utf-8'))
        model=joblib.load(ARTIFACTS/'model.joblib')
        table=supervised_table(self.frame)
        train=table.loc[split_masks(table)['train']]
        numeric=FEATURE_SETS[metadata['feature_set']]
        np.testing.assert_allclose(model['preprocess'].named_transformers_['numeric']['imputer'].statistics_,train[numeric].median())
        self.assertFalse(set(metadata['features']) & {'pm25_target','target_time','label','No'})
        features=build_features(self.frame,'Aotizhongxin','2016-05-01 09:00')
        result=predict_pollution_class(features)
        self.assertAlmostEqual(sum(result['probabilities'].values()),1)
        self.assertTrue(all(0<=p<=1 for p in result['probabilities'].values()))
        expected=model.predict_proba(pd.DataFrame([features]))[0]
        for k,p in zip(model.classes_,expected):self.assertAlmostEqual(result['probabilities'][CLASS_NAMES[int(k)]],p)
        for bad in [dict(features,station='unknown'),dict(features,**{'PM2.5':-1}),{k:v for k,v in features.items() if k!='TEMP'}]:
            with self.assertRaises(ValueError):predict_pollution_class(bad)

if __name__=='__main__':unittest.main()
