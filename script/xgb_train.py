# -*- coding: utf-8 -*
import pandas as pd
import xgboost as xgb
from sklearn import cross_validation, metrics
import cPickle as pickle

#read data
tr = pd.read_csv("train_test_data/tr.csv")
va = pd.read_csv("train_test_data/va.csv")

y = tr.is_risk
X = tr.drop(['is_risk'], axis=1)
val_y = va.is_risk
val_X = va.drop(['is_risk'], axis=1)

xgb_val   = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)

params = {
    'booster': 'gbtree',
	'objective': 'binary:logistic',
	'eta': 0.1,
	'max_depth': 4,
	'gamma': 0.1,
	'subsample': 0.8,
	'colsample_bytree': 0.8,
	'min_child_weight': 1,
	'scale_pos_weight': 1,
	'seed': 29,
	'nthread': 7
}

plst = list(params.items())
num_rounds = 12000
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

def f1_score(preds, dtrain):
    labels = dtrain.get_label()
	return metrics.fbeta_score(labels, preds, beta=0.1)

model = xgb.train(plst, xgb_train, num_rounds, watchlist, earle_stopping_rounds=100, verbose_eval=1, feval=f1_score, maximize=True)

f1 = file('model/xgb_model.pkl', 'wb')
pickle.dump(model, f1, True)
