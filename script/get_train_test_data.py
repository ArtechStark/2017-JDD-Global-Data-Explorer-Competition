# -*- coding: utf-8 -*-

import pandas as pd

id_train  = pd.read_csv("data/id_train.csv")
ip_train  = pd.read_csv("data/ip_train.csv")
dev_train = pd.read_csv("data/dev_train.csv")

train     = pd.concat([id_train, ip_train, dev_train], axis=1)
tr        = train.loc(train['month']<5)
val       = train.loc(train['month']==5)
tr.to_csv("train_test_data/tr.csv", index=False)
val.to_csv("train_test_data/val.csv", index=False)

id_test   = pd.read_csv("data/id_test.csv")
ip_test   = pd.read_csv("data/ip_test.csv")
dev_test  = pd.read_csv("data/dev_test.csv")

test      = pd.concat([id_test, ip_test, dev_test], axis=1)
test.to_csv("train_test_data/test.csv", index=False)