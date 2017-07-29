#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from subprocess import run, PIPE, DEVNULL, STDOUT

def seg2df(path):
    with open(path) as f:
        lines = [n for n, r in enumerate(f.read().splitlines()) if r.startswith(';;')]
    df = pd.read_csv(path, skiprows=lines, delimiter=' ', header=None,
            usecols=[2,3,7], names=['start', 'dur', 'spkr'])
    df.spkr = df.spkr.str[1:].astype(np.int16)
    #f = lambda x: timedelta(seconds=x/100.)
    df[:] = df.sort_values('start').reset_index(drop=True)
    df['lbl'] = df.spkr
    df['gen'] = [0]*len(df)
    return df


def lbl2df(path, start=10, filemap=False):
    lbls = [n for n in os.listdir(path) if n.endswith('.lbl')]
    lbls.sort()
    dfs = []
    cls = start
    for n, i in enumerate(lbls):
        df = pd.read_csv(i, delimiter=' ', index_col=0)
        df['file'] = [n]*len(df) if filemap else [i.replace('lbl', 'mp3')]*len(df)
        cmap = {n: i + cls for i, n in enumerate(df.loc[df.lbl > 0, 'lbl'].unique())}
        for n in range(-2, 1): cmap[n] = n
        df['cls'] = df.lbl.map(cmap)
        cls += len(cmap)
        dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)

    if filemap:
        return {n: i for n, i in enumerate(lbls)}, dfs
    else:
        return dfs


def cplay(df):
    if type(df) != pd.core.series.Series:
        for _, i in df.iterrows():
            print(i.name, i.lbl, i.cls, i.file, i.dur*0.01)
            run(['play', i.file, 'trim', str(i.start*0.01), str(i.dur*0.01)],
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)
    else:
        print(df.name, df.lbl, df.cls, df.file, df.dur*0.01)
        run(['play', df.file, 'trim', str(df.start*0.01), str(df.dur*0.01)],
                stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)


def play(seg, df):
    for n, i in df.iterrows():
        print(n, i.spkr, i.start*0.01/3600, i.dur*0.01)
        run(['play', seg, 'trim', str(i.start*0.01), str(i.dur*0.01)], stdout=DEVNULL)


def splay(seg, df, *args):
    for spk in args:
        play(seg, df.loc[df.spkr == spk])


def lbl2seg(path):
    name = path.split('.')[0]
    df = pd.read_csv(path, delimiter=' ', index_col=0)

    df.index = [name]*len(df)
    gmap = {-1: 'U', 0: 'M', 1: 'F'}
    df.gen = df.gen.map(gmap)
    lmap = {n: 'S' + str(n) for n in df.lbl.unique()}
    df.lbl = df.lbl.map(lmap)
    df[['start', 'dur']] = df[['start', 'dur']].astype(str)

    df['ch'] = '1'
    df['env'] = 'S'
    df['typ'] = 'U'
    df = df[['ch', 'start', 'dur', 'gen', 'env', 'typ', 'lbl']]

    return df


def calc_der(lbl):
    from scipy.optimize import linear_sum_assignment as lsa

    # the labels -2 and -1, and 0 are reserved for overlapping speech, non-
    # speech, and silence, respectively, in the reference. overlapping speech,
    # however, is not considered during the hypothesis formation, and hence is
    # excluded from the speaker mapping.
    sp = np.sort(lbl.lbl.unique())
    lmap = {n: i for i, n in enumerate(sp[sp > -2])}
    for n in range(-2, 0): lmap[n] = n
    smap = {n: i for i, n in enumerate(lbl.spkr.unique())}

    lbl['ref'] = lbl.lbl.map(lmap)
    lbl['hyp'] = lbl.spkr.map(smap)
    spk = lbl.loc[lbl.lbl > -2]

    # obtain 1-1 mapping between speaker (hyp) and label (ref)
    # group by label first (ref), then by speaker (hyp)
    freq = spk.groupby([spk['ref'], spk['hyp']]).dur.sum()
    # a normal dictionary is insufficient as both 1-n and n-1 are possible
    # so instead create a N*M sparse matrix containing durations of hyp label
    # N and ref label M, representing a weighted bipartite graph.
    cmat = np.zeros((len(spk.ref.unique()), len(spk.hyp.unique())))
    cmat[freq.index.labels] = freq.values.flat
    # with the constraint that we can only use ONE of each N and M, optimize
    # for largest global sum using the hungarian algorithm for maximum weight
    # matching in bipartite graphs. as linear sum assignment calculates the
    # minimum, we must first inverse the costs after adding a small non-zero
    # value to avoid division by zero.
    ref, hyp = lsa(1./(cmat + 0.1))

    # calculate diarization error rate (not including overlapping segments)
    return 1.0 - (cmat[ref, hyp].sum() / spk.dur.sum())


def make_ubm(out, path='/home/cilsat/data/speech/rapat', min_dur=15000, min_spk=3):
    # concat all lbl files and make unique clusters
    dfs = lbl2df(path, 1)
    # get all clusters that are not the first min_spk speakers in each file
    # the assumption here is that the first min_spk speakers are repeated across
    # meetings
    spk = dfs.loc[dfs.groupby(dfs.file).apply(lambda x: x.loc[x.lbl.isin(x.lbl.unique()[min_spk:])]).index.get_level_values(1)]
    spk = spk.loc[spk.lbl > 1]
    # find clusters that have at least min_dur seconds of speech and get them
    segs = []
    for n in spk.cls.unique():
        dfc = spk.loc[spk.cls == n]
        cum = dfc.dur.cumsum()
        if cum.max() > min_dur:
            df = dfc.loc[:cum.loc[cum > min_dur].index[0]]
            # get start and end of segments in samples
            trims = np.dstack((df.start.values, (df.start + df.dur).values)).flatten()*160
            # write segments to file
            run(['sox', os.path.join(path, df.file.iloc[0]), os.path.join(path, 'ubm/c'+str(n).zfill(3)+'.wav'), 'trim'] + ['='+str(n)+'s' for n in trims], stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)

    return pd.concat(segs)


if __name__ == "__main__":
    path = sys.argv[1]
    df = seg2df(path)
    fix_lbl(path, df)
