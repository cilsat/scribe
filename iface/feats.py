#!/usr/bin/python2

# methods for feature computation and alignment at the frame and utterance
# levels.

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
    return pd.concat((a, df_d, df_dd), axis=1).astype(np.float16)


def count_phones(args):
    pg, mg = args
    lpg = len(pg)
    lmg = len(mg)
    ali_idx = [x for n in [[i]*pg.dur[i] for i in xrange(lpg - 1)] for x in n]
    ali_idx.extend([ali_idx[-1] + 1 for _ in xrange(lmg - len(ali_idx))])
    mg['ord'] = np.array(ali_idx, dtype=np.uint16)
    return mg


# parallelize functions applied to dataframe groups
def apply_parallel(func, args=(), n_proc=cpu_count()):
    p = Pool(n_cpu)
    ret = p.map(func, args)
    p.close()
    p.terminate()
    return pd.concat(ret)


# computes delta features in place
def compute_deltas(df_mfcc, n=2):
    # group features by utterance and compute deltas/delta-deltas
    mg = df_mfcc.groupby(df_mfcc.index)
    w = np.array([n for n in xrange(-n, n+1)])
    d = 1./np.sum([2*i*i for i in xrange(1, n+1)])
    df_deltas = apply_parallel(roll_delta, [(g, n, w, d) for _,g in mg])

    # fix column names
    n_feats = len(df_mfcc.columns)
    c1 = [c for c in df_mfcc.columns]
    c2 = ['d_' + c for c in df_deltas.columns[n_feats:2*n_feats]]
    c3 = ['dd_' + c for c in df_deltas.columns[2*n_feats:]]
    df_deltas.columns = c1 + c2 + c3
    return df_deltas


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
    return apply_parallel(count_phones, [(g, mg.get_group(n)) for n, g in pg])


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
def compute_utt_feats(df_ali):
    dfg_ali = df_ali.groupby([df_ali.index, df_ali['ord']])

    print("computing segment durations and delta segment durations")
    dur = pd.Series(dfg_ali.eng.count(), name='dur', dtype=np.int16)
    ddur = dur.groupby(dur.index.get_level_values(0)).diff()
    dddur = ddur.groupby(ddur.index.get_level_values(0)).diff()
    ddur.fillna(0, inplace=True)
    dddur.fillna(0, inplace=True)

    print("computing per segment MFCC means")
    means = dfg_ali.mean()

    print("computing delta segment MFCC means")
    dmeans = means.groupby(means.index.get_level_values(0)).diff()
    dmeans.fillna(0, inplace=True)

    print("computing delta delta segment MFCC means")
    ddmeans = dmeans.groupby(dmeans.index.get_level_values(0)).diff()
    ddmeans.fillna(0, inplace=True)

    ddur.name = 'ddur'
    dddur.name = 'dddur'
    means.columns = ['avg_' + c for c in means.columns]
    dmeans.columns = ['d_' + c for c in dmeans.columns]
    ddmeans.columns = ['dd_' + c for c in ddmeans.columns]

    feats = pd.concat((dur, means, ddur.astype(np.int16), dmeans, dddur.astype(np.int16), ddmeans), axis=1)

    print("computing per utterance means and variances")
    dfg_feat = feats.groupby(feats.index.get_level_values(0))
    utts = pd.concat((dfg_feat.mean(), dfg_feat.var()), axis=1)
    cols = feats.columns
    utts.columns = ['avg_'+c for c in cols] + ['var_'+c for c in cols]

    return utts

