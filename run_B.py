import re
import os
import sklearn
import json
import pandas as pd
import warnings
import multiprocessing
import toad

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


warnings.filterwarnings('ignore')

##############
# 读取数据
# 拼接数据
##############

if os.path.exists('data.pkl'):
    data = pd.read_pickle('data.pkl')
else:
    train = pd.read_excel('train.xlsx')
    test = pd.read_excel('test_B榜.xlsx')

    data = pd.concat([train, test]).reset_index(drop=True)


##############
# 数据预处理
# 编码、缺失值等
# 数值特征-2以对准
##############

ff = [i for i in data.columns if i not in ['LABEL', 'CUST_UID']]
cat_f = ['MON_12_CUST_CNT_PTY_ID',  'WTHR_OPN_ONL_ICO',  'LGP_HLD_CARD_LVL', 'NB_CTC_HLD_IDV_AIO_CARD_SITU']
num_f = []
for f in tqdm(ff):
    data[f] = data[f].fillna(-2)
    data[f] = data[f].astype('str')
    data[f] = data[f].apply(lambda x: x.replace('?', '-1'))

    if f not in cat_f:
        data[f] = data[f].astype('float')
        data[f] = data[f].replace(-1, np.nan)
    else:
        if f == 'MON_12_CUST_CNT_PTY_ID':
            lb = LabelEncoder()
            data[f] = lb.fit_transform(data[f])
        else:
            grade_dict = {'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1}
            data[f] = data[f].map(grade_dict)
            data[f] = data[f].replace(-1, np.nan)

    data[f] -= 2
    if data[f].max() > 1000000:
        num_f.append(f)


##############
# 可能是毒特的feature
# 所有CUR以及时间相关等
##############

cur_f_list = [i for i in data.columns if 'CUR' in i] + ['OPN_TM', 'REG_DT',  'REG_CPT',
                                                          'COR_KEY_PROD_HLD_NBR',
                                                          'WTHR_OPN_ONL_ICO',
                                                          'NB_RCT_3_MON_LGN_TMS_AGV',
                                                          'AGN_AGR_LATEST_AGN_AMT'
                                                       ]

##############
# 分离训练测试
##############

train = data[~data['LABEL'].isna()].reset_index(drop=True)
test = data[data['LABEL'].isna()].reset_index(drop=True)

features = [i for i in train.columns if i not in ['LABEL', 'CUST_UID',
                                                 ] + cur_f_list]
y = train['LABEL']
print("Train files: ", len(train), "| Test files: ", len(test), "| Feature nums", len(features))

import gc
del data
gc.collect

def train_model(X_train, X_test, features, y, seed=2021, save_model=False):
    """
    训练lgb模型
    """
    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    score_list = []
    params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 50,
        'verbose': -1,
        'seed': seed,
        'bagging_seed': seed,
        'feature_fraction_seed': seed,
        'early_stopping_rounds': 100,

    }
    oof_lgb = np.zeros(len(X_train))
    predictions_lgb = np.zeros((len(X_test)))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
        trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,
        )

        oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration) / 5
        feat_imp_df['imp'] += clf.feature_importance() / 5
        score_list.append(roc_auc_score(y.iloc[val_idx], oof_lgb[val_idx]))
        if save_model:
            clf.save_model(f'model_{fold_}.txt')

    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("AUC mean: {}".format(np.mean(score_list)))
    return feat_imp_df, oof_lgb, predictions_lgb


##############
# 模型多种子训练
##############

seeds = [666, 888, 999]

pred = []
oof = []
for seed in seeds:
    feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y, seed)
    pred.append(predictions_lgb)
    oof.append(oof_lgb)


##############
# 均值融合
# 生成提交文件
##############

test['label'] = np.mean(pred, axis=0)
test[['CUST_UID', 'label']].to_csv('b_sub_8524.txt', index=False, header=None, sep=' ')
print(test[['CUST_UID', 'label']].head())
len(test)
