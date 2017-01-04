#!/usr/bin/python2

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count


# calculate delta and delta-delta features in an utterance
# delta computation modified from https://github.com/jameslyons/
# python_speech_features/blob/master/python_speech_features/base.py
def roll_delta(args):
    a, n, w, d = args

    def delta(df):
        df_ret = pd.concat((pd.DataFrame([df.iloc[0] for _ in xrange(n)]),
            df, pd.DataFrame([df.iloc[-1] for _ in xrange(n)]))).astype(np.float16)
        df_ret[:] = df_ret.rolling(window=2*n+1, center=True).apply(lambda x: np.sum(w.T*x, axis=0)*d)
        df_ret.dropna(inplace=True)
        return df_ret

    df_d = delta(a)
    df_dd = delta(df_d)
    return pd.concat((df_d, df_dd), axis=1).astype(np.float16)

def apply_parallel(df_group, func, args):
    args = [[g] + args for _, g in df_group]
    p = Pool(cpu_count())
    ret = p.map(func, args)
    p.close()
    p.terminate()
    p.join()
    return pd.concat(ret).astype(np.float16)

# computes delta features in place
def compute_deltas(df_mfcc, n=2):
    # group features by utterance and compute deltas/delta-deltas
    mg = df_mfcc.groupby(df_mfcc.index)
    w = np.array([n for n in xrange(-n, n+1)])
    d = 1./np.sum([2*i*i for i in xrange(1, n+1)])
    df_deltas = apply_parallel(mg, roll_delta, [n, w, d]) 
    n_feats = len(df_mfcc.columns)
    df_deltas.columns = ['d_' + c for c in df_deltas.columns[:n_feats]] + ['dd_' + c for c in df_deltas.columns[n_feats:]]
    return pd.concat((df_mfcc, df_deltas)).astype(np.float16)

# align raw mfcc dataframe to raw phone dataframe
# return raw mfcc frames with phone segment label
def ali_mfcc_phon(df_mfcc, df_phon):
    # use the minimal subset of utt/files contained in both dataframes
    dif = set(df_mfcc.index) ^ set(df_phon.index)
    df_mfcc.drop(dif, inplace=True)
    df_phon.drop(dif, inplace=True)

    # align features to phone boundaries
    pg_pos = df_phon.groupby(df_phon.index)['pos']

