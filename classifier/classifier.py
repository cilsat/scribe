#!/usr/bin/python3

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from multiprocessing import cpu_count

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

