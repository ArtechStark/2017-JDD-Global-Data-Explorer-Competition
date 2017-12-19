        # -*- coding: utf-8 -*
import pandas as pd
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import cPickle as pickle

#read data
df     = pd.read_csv("train_test_data/test.csv")
rowkey = list(df['rowkey'])

feas   = df.drop(['rowkey'], axis=1).copy()

#load model and predict
f1     = file("model/xgb_model.pkl")
model  = pickle.load(f1)
preds  = model.predict(feas)

dic    = {'rowkey':rowkey, 'is_risk':preds}
cols   = ['rowkey', 'is_risk']
result = pd.DataFrame(dic)
result = result.ix[:, cols]

#save to disk
result.to_csv("result/xgb_res.csv", index=False, header=False, encoding='utf8')