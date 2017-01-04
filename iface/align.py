#!/usr/bin/python2

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count


# calculate delta and delta-delta features in an utterance
# delta computation modified from https://github.com/jameslyons/
# python_speech_features/blob/master/python_speech_features/base.py
def roll_delta(args):
    a, n, w, d = args
    delta = pd.concat((pd.DataFrame([a.iloc[0] for _ in xrange(n)]),
        a, pd.DataFrame([a.iloc[-1] for _ in xrange(n)])))
    delta[:] = delta.rolling(window=2*n+1, center=True).apply(lambda x: np.sum(w.T*x, axis=0)*d)
    delta.dropna(inplace=True)
    return delta

def apply_parallel(dfg, func, args):
    args = [[g] + args for _, g in dfg]
    p = Pool(cpu_count())
    ret = p.map(func, args)
    p.close()
    p.terminate()
    p.join()
    return pd.concat(ret)

# align raw mfcc dataframe to raw phone dataframe
# return raw mfcc frames with phone segment label
def ali_mfcc_phon(df_mfcc, df_phon):
    # use the minimal subset of utt/files contained in both dataframes
    dif = set(df_mfcc.index) ^ set(df_phon.index)
    df_mfcc.drop(dif, inplace=True)
    df_phon.drop(dif, inplace=True)

    # group features by utterance and apply delta function on each utterance
    mg = df_mfcc.groupby(df_mfcc.index)
    n = 2
    w = np.array([n for n in xrange(-n, n+1)])
    d = 1./np.sum([2*i*i for i in xrange(1, n+1)])
    deltas = apply_parallel(mg, roll_delta, [n, w, d]) 
    deltas.columns = ['d_' + col for col in delta.columns]

    return deltas
