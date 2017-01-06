#!/usr/bin/python2

# methods for feature computation and alignment at the frame, segment, and
# utterance levels.

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import zip_longest

# parallelize functions applied to dataframe groups
# it should be noted that this is *very inefficient* for multi-indexed groups
# i.e. dataframes grouped on more than one axis due to, i believe, poor pickle
# performance
def apply_parallel(func, args):
    with Pool(cpu_count()) as p:
        ret = p.starmap(func, args)
    return pd.concat(ret)


# calculate delta and delta-delta features in an utterance dataframe
# delta computation modified from https://github.com/jameslyons/
# python_speech_features/blob/master/python_speech_features/base.py
def roll_delta(dfg, n, w, d):
    def delta(df):
        df_ret = pd.concat((pd.DataFrame([df.iloc[0] for _ in range(n)]),
            df, pd.DataFrame([df.iloc[-1] for _ in range(n)]))).astype(np.float16)
        df_ret[:] = df_ret.rolling(window=2*n+1, center=True).apply(lambda x: np.sum(w.T*x, axis=0)*d)
        df_ret.dropna(inplace=True)
        return df_ret

    df_d = delta(dfg)
    df_dd = delta(df_d)
    return pd.concat((dfg, df_d, df_dd), axis=1).astype(np.float16)


# computes delta features in place
def compute_deltas(df_mfcc, n=2):
    # group features by utterance and compute deltas/delta-deltas
    mg = df_mfcc.groupby(df_mfcc.index)
    w = np.array([n for n in range(-n, n+1)])
    d = 1./np.sum([2*i*i for i in range(1, n+1)])
    df_deltas = apply_parallel(roll_delta, [(g, n, w, d) for _, g in mg])

    # fix column names
    n_feats = len(df_mfcc.columns)
    c1 = [c for c in df_mfcc.columns]
    c2 = ['d_' + c for c in df_deltas.columns[n_feats:2*n_feats]]
    c3 = ['dd_' + c for c in df_deltas.columns[2*n_feats:]]
    df_deltas.columns = c1 + c2 + c3
    return df_deltas


# align MFCC frames in one dataframe to phone segments in another dataframe
# creates a new column in the MFCC dataframe indicating phone count, i.e. the
# phone order of a particular frame within an utterance
def count_phones(mg, pg):
    lpg = len(pg)
    lmg = len(mg)
    ali_idx = [x for n in [[i]*pg.dur[i] for i in range(lpg - 1)] for x in n]
    ali_idx.extend([ali_idx[-1] + 1 for _ in range(lmg - len(ali_idx))])
    mg['ord'] = np.array(ali_idx, dtype=np.uint16)
    return mg


# align raw mfcc dataframe to raw phone dataframe
# return raw mfcc frames with phone segment label
def align_phones(df_mfcc, df_phon):
    # use the minimal subset of utt/files contained in both dataframes
    # drop utterances with only 1 phone
    dif = set(df_mfcc.index) ^ set(df_phon.index)
    drop = set(df_phon.loc[df_phon.groupby(df_phon.index).dur.count() == 1].index)
    drop |= dif
    df_mfcc.drop(drop, inplace=True)
    df_phon.drop(drop, inplace=True)

    # construct indices of frames from phone alignments
    pg = df_phon.groupby(df_phon.index)
    mg = df_mfcc.groupby(df_mfcc.index)
    return apply_parallel(count_phones, [(mg.get_group(n), g) for n, g in pg])


# process all alignment files in a given directory
# MFCC files are assumed to begin with 'mfcc' and phone alignments files with
# 'phon'. returns dataframe indexed by utterance name containing mfccs,
# delta/delta2s, and phone order within utterance
def process_path(path, output='ali.hdf'):
    import iface, os
    import cPickle as pickle

    print("parsing files")
    mfcc_args = []
    phon_args = []
    for f in os.listdir(path):
        if f.startswith('mfcc'):
            mfcc_args.append((os.path.join(path, f), 'mfcc'))
        elif f.startswith('phon'):
            phon_args.append((os.path.join(path, f), 'phon'))

    mfccs = apply_parallel(iface.ali2df, mfcc_args)
    phons = apply_parallel(iface.ali2df, phon_args)
    print(mfccs.info())
    print(phons.info())

    print("\ncomputing deltas")
    mfccs = compute_deltas(mfccs)
    print(mfccs.info())

    print("\naligning phones")
    mfccs = align_phones(mfccs, phons)
    print(mfccs.info())

    mfccs.to_hdf(output, 'ali')
    phons.to_hdf(output, 'phon')


# compute segment-level features for utterance classification from a phone
# aligned mfcc dataframe. features include per segment frame averages, variances
# and their deltas.
def compute_seg_feats(df_ali):

    def calc_durs(dfg):
        dur = pd.Series(dfg.eng.count(), name='dur', dtype=np.int16)
        d_dur = dur.groupby(dur.index.get_level_values(0)).diff()
        dd_dur = d_dur.groupby(d_dur.index.get_level_values(0)).diff()
        d_dur.fillna(0, inplace=True)
        dd_dur.fillna(0, inplace=True)
        d_dur.name = 'd_dur'
        dd_dur.name = 'dd_dur'
        return pd.concat((dur, d_dur, dd_dur), axis=1).astype(np.int16)

    def calc_means(dfg):
        print("computing per segment MFCC means")
        mean = dfg.mean()

        print("computing delta segment MFCC means")
        d_mean = mean.groupby(mean.index.get_level_values(0)).diff()
        d_mean.fillna(0, inplace=True)

        print("computing delta delta segment MFCC means")
        dd_mean = d_mean.groupby(d_mean.index.get_level_values(0)).diff()
        dd_mean.fillna(0, inplace=True)

        mean.columns = ['avg_' + c for c in mean.columns]
        d_mean.columns = ['d_' + c for c in d_mean.columns]
        dd_mean.columns = ['dd_' + c for c in dd_mean.columns]
        return pd.concat((mean, d_mean, dd_mean), axis=1)

    dfg_ali = df_ali.groupby([df_ali.index, df_ali['ord']])

    print("computing segment durations and delta segment durations")
    durs = calc_durs(dfg_ali)
    print("computing segment means and delta segment durations")
    means = calc_means(dfg_ali)

    df_seg = pd.concat((durs, means), axis=1)
    return df_seg


def compute_utt_feats(df_seg):
    print("computing per utterance means and variances")
    dfg_feat = df_seg.groupby(df_seg.index.get_level_values(0))
    df_utt = pd.concat((dfg_feat.mean().astype(np.float16), dfg_feat.var().astype(np.float16)), axis=1)
    cols = df_seg.columns
    df_utt.columns = ['avg_'+c for c in cols] + ['var_'+c for c in cols]

    return df_utt

