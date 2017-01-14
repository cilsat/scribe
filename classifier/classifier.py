#!/usr/bin/python3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from multiprocessing import cpu_count
import os

def score(clf, df_spon, df_read, sample=True):
    if sample:
        data = pd.concat((df_spon, df_read.sample(len(df_spon))))
        lbl = [True]*len(df_spon) + [False]*len(df_spon)
    else:
        data = pd.concat((df_spon, df_read))
        lbl = [True]*len(df_spon) + [False]*len(df_read)

    scores = cross_val_score(clf, data, lbl, cv=10, n_jobs=cpu_count())
    return scores

def train(clf, df_spon, df_read, sample=True):
    if sample:
        data = pd.concat((df_spon, df_read.sample(len(df_spon))))
        lbl = [True]*len(df_spon) + [False]*len(df_spon)
    else:
        data = pd.concat((df_spon, df_read))
        lbl = [True]*len(df_spon) + [False]*len(df_read)

    clf.fit(data, lbl)
    try:
        weights = pd.Series(clf.feature_importances_, index=df_spon.columns).sort_values()
    except:
        pass

# needs fold information in dataframe
def run_expr(spon_path, read_path, res_path, kf=10, sample=False):
    df_spon = pd.read_hdf(spon_path, 'utt')
    df_spon['lbl'] = [True]*len(df_spon)
    df_read = pd.read_hdf(read_path, 'utt')
    df_read['lbl'] = [False]*len(df_read)

    # do for each fold
    for k in range(kf):
        clf = RandomForestClassifier(n_estimators=25, max_depth=None, min_samples_split=10, n_jobs=cpu_count())
        spon_train = df_spon.loc[df_spon.fold != k]
        read_train = df_read.loc[df_read.fold != k]
        if sample:
            read_train = read_train.sample(len(spon_train))

        train = pd.concat((spon_train, read_train)).drop('fold', axis=1)
        clf.fit(train.drop('lbl', axis=1), train.lbl)

        spon_test = df_spon.loc[df_spon.fold == k]
        spon_hyp = clf.predict(spon_test.drop(['lbl', 'fold'], axis=1))
        spon_miss = spon_test.loc[spon_test.lbl != spon_hyp]
        spon_res = pd.concat((spon_test.lbl, pd.Series(spon_hyp, index=spon_test.index, name='hyp')), axis=1)
        spon_res.to_csv(os.path.join(res_path, 'full-spon-'+str(k)))

        with open(os.path.join(res_path, 'miss-spon-'+str(k)), 'w') as f:
            [f.write(res + '\n') for res in list(spon_miss.index)]

        read_test = df_read.loc[df_read.fold == k]
        read_hyp = clf.predict(read_test.drop(['lbl', 'fold'], axis=1))
        read_miss = read_test.loc[read_test.lbl != read_hyp]
        read_hyp = pd.concat((read_test.lbl, pd.Series(read_hyp, index=read_test.index, name='hyp')), axis=1)
        read_hyp.to_csv(os.path.join(res_path, 'full-read-'+str(k)))

        with open(os.path.join(res_path, 'miss-read-'+str(k)), 'w') as f:
            [f.write(res + '\n') for res in list(read_miss.index)]

