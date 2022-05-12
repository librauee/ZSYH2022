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
    test = pd.read_excel('test_A榜.xlsx')

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
# 特征工程
# 1. 偏离值特征
# 2. 数值和类别特征交叉
# 3. 加减乘除交叉
##############

for group in tqdm(cat_f):
    for feature in ff:
        if feature not in cat_f:
            tmp = data.groupby(group)[feature].agg([sum, min, max, np.mean]).reset_index()
            tmp = pd.merge(data, tmp, on=group, how='left')
            data['{}-mean_gb_{}'.format(feature, group)] = data[feature] - tmp['mean']
            data['{}-min_gb_{}'.format(feature, group)] = data[feature] - tmp['min']
            data['{}-max_gb_{}'.format(feature, group)] = data[feature] - tmp['max']
            data['{}/sum_gb_{}'.format(feature, group)] = data[feature] / tmp['sum']

for i in tqdm(range(len(num_f))):
    for j in range(i + 1, len(num_f)):
        for cat in cat_f[1:]:
            f1 = ff[i]
            f2 = ff[j]
            data[f'{f1}_{f2}_log_{cat}'] = (np.log1p(data[f1]) - np.log1p(data[f2])) * data[cat]
            data[f'{f1}+{f2}_log_{cat}'] = (np.log1p(data[f1]) + np.log1p(data[f2])) * data[cat]
            data[f'{f1}*{f2}_log_{cat}'] = (np.log1p(data[f1]) * np.log1p(data[f2])) * data[cat]
            data[f'{f1}/{f2}_log_{cat}'] = (np.log1p(data[f1]) / np.log1p(data[f2])) * data[cat]
            data[f'{f2}/{f1}_log_{cat}'] = (np.log1p(data[f2]) / np.log1p(data[f1])) * data[cat]

            data[f'{f1}_{f2}_log_{cat}_'] = (np.log1p(data[f1]) - np.log1p(data[f2])) / data[cat]
            data[f'{f1}+{f2}_log_{cat}_'] = (np.log1p(data[f1]) + np.log1p(data[f2])) / data[cat]
            data[f'{f1}*{f2}_log_{cat}_'] = (np.log1p(data[f1]) * np.log1p(data[f2])) / data[cat]
            data[f'{f1}/{f2}_log_{cat}_'] = (np.log1p(data[f1]) / np.log1p(data[f2])) / data[cat]
            data[f'{f2}/{f1}_log_{cat}_'] = (np.log1p(data[f2]) / np.log1p(data[f1])) / data[cat]


for i in tqdm(range(len(ff))):
    for j in range(i + 1, len(ff)):
        f1 = ff[i]
        f2 = ff[j]
        data[f'{f1}_{f2}'] = data[f1] - data[f2]
        data[f'{f1}+{f2}'] = data[f1] + data[f2]
        data[f'{f1}*{f2}'] = data[f1] * data[f2]
        data[f'{f1}/{f2}'] = data[f1] / data[f2]
        data[f'{f2}/{f1}'] = data[f2] / data[f1]


##############
# 减少内存使用
##############

def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in tqdm(features):
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

data = reduce_mem_usage(data)


##############
# 分离训练测试
##############

train = data[~data['LABEL'].isna()].reset_index(drop=True)
test = data[data['LABEL'].isna()].reset_index(drop=True)

features = [i for i in train.columns if i not in ['LABEL', 'CUST_UID',
                                                 ]]
y = train['LABEL']
print("Train files: ", len(train), "| Test files: ", len(test), "| Feature nums", len(features))


def train_model(X_train, X_test, features, y, seed=2021, save_model=False):

    feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    score_list = []
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
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
        'max_bin': 255,
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
# 模型全特征训练
# 得到特征重要性排名
##############
seeds = [2021]

pred = []
oof = []
for seed in seeds:
    feat_imp_df, oof_lgb, predictions_lgb = train_model(train, test, features, y, seed)
    pred.append(predictions_lgb)
    oof.append(oof_lgb)


np.save('score_dict_a.npy', off_score_dict)
feat_imp_df.to_pickle('feature_imp_a.pkl')


##############
# 遍历特征组合
# 寻找较优的验证分数模型
##############

from collections import defaultdict

off_score_dict = defaultdict(int)

for i in range(201, 501):

    features2 = feat_imp_df.sort_values(['imp'])[-i:]['feat'].to_list()
    seeds = [2021]
    pred = []
    oof = []
    for seed in seeds:
        _, oof_lgb, predictions_lgb = train_model(train, test, features2, y, seed)
        pred.append(predictions_lgb)
        oof.append(oof_lgb)

    score_ = roc_auc_score(y, np.mean(oof, axis=0))
    if score_ > 0.953:
        off_score_dict[i] = score_


##############
# 均值融合
# 生成提交文件
##############

score_dict = np.load('score_dict_a.npy', allow_pickle=True).item()
feat_imp_df = pd.read_pickle('feature_imp_a.pkl')

pred = []
oof = []
for k, v in tqdm(score_dict.items()):
    if v > 0.9532:
        features2 = feat_imp_df.sort_values(['imp'])[-k:]['feat'].to_list()
        _, oof_lgb, predictions_lgb = train_model(train, test, features2, y, seed=2021)
        pred.append(predictions_lgb)
        oof.append(oof_lgb)

test['label'] = np.mean(pred, axis=0)
test[['CUST_UID', 'label']].to_csv('sub.txt', index=False, header=None, sep=' ')
print(test[['CUST_UID', 'label']].head())
len(test)