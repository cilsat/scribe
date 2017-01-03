#!/usr/bin/python2

import pandas as pd
import numpy as np
import multiprocessing as mp


# align raw mfcc dataframe to raw phone dataframe
# return raw mfcc frames with phone segment label
def ali_mfcc_phon(df_mfcc, df_phon):
    # use the minimal subset of utt/files contained in both dataframes
    dif = set(df_mfcc.index) ^ set(df_phon.index)
    df_mfcc.drop(dif, inplace=True)
    df_phon.drop(dif, inplace=True)

    # calculate delta and delta-delta
    # window size for delta and delta-delta feature calculation
    n = 5
    denom = np.sum([2*i*i for i in xrange(1, n+1)])
    def delta(feats):
        num_frames = len(feats)
        feats = np.concatenate(([feats[0] for i in xrange(n)], feats, [feats[-1] for i in xrange(n)]))

    mg = df_mfcc.groupby(df_mfcc.index)
    deltas = mg.transform(lambda x: delta(x, del_win))
