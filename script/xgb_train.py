# -*- coding: utf-8 -*

import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
import cPickle as pickle

#read data
tr = pd.read_csv("train_test_data/tr.csv")
va = pd.read_csv("train_test_data/va.csv")

target = 'is_risk'
predictors = [x for x in tr.columns if x not in [target]]
vaPredictors = [x for x in va.columns if x not in [target]]

def modelfit(alg, dtrain, dtest, predictors, testPredictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    #fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    #predict in train data
    dtrain_predictions = alg.predict(dtrain[predictors])

    #predict on val data
    dtest_predictions = alg.predict(dtest[testPredictors])

    # model report
    print "\nModel Report"
    s = metrics.fbeta_score(dtrain[target].values, dtrain_predictions, beta=0.1)
    print ("The train score is: %f" % s)
    s = metrics.fbeta_score(dtest[target].values, dtest_predictions, beta=0.1)
    print ("Tha test score is: %f" % s)

    #save feature score
    fea_score = alg.booster().get_fscore()
    fea_score = sorted(fea_score.items(), key=lambda x:x[1],reverse=True)
    fs        = []
    for (key,value) in fea_score:
        fs.append("{0},\t\t{1}\n".format(key,value))
    with open('feature_score/xgb_feature_score.csv','w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

    #save model
    f1 = file('model/xgb_model.pkl', 'wb')
    pickle.dump(alg, f1, True)

xgb1 = XGBClassifier(
        objective='binary:logistic',
        learning_rate =0.1,
        n_estimators=10000,
        max_depth=4,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        seed=29,
        nthread=2
)

modelfit(xgb1, tr, va, predictors, vaPredictors)