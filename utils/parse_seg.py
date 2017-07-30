#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from subprocess import run, PIPE, DEVNULL, STDOUT

dsp = ['gain', '-6', 'highpass', '120']


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
            print(i.name, i.cls, i.file, i.dur)
            run(['play', i.file, 'trim', str(i.start*160)+'s', str(i.dur*160)+'s'] + dsp,
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)
    else:
        print(df.name, df.cls, df.file, df.dur*160)
        run(['play', df.file, 'trim', str(df.start*160)+'s', str(df.dur*160)+'s'] + dsp,
                stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)


def play(seg, df):
    for n, i in df.iterrows():
        print(n, i.spkr, i.start*0.01/3600, i.dur*0.01)
        run(['play', seg, 'trim', str(i.start*0.01), str(i.dur*0.01)] + dsp,
                stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)


def splay(seg, df, *args):
    for spk in args:
        play(seg, df.loc[df.spkr == spk])


def lbl2seg(name, path=True, s='lbl'):
    if path:
        name = name.split('.')[0]
        df = pd.read_csv(path, delimiter=' ', index_col=0)
    else: df = name

    df.index = df.file.str[:4] if 'file' in df.columns else [name]*len(df)
    gmap = {-1: 'U', 0: 'M', 1: 'F'}
    df['gen'] = df['gen'].map(gmap)
    fill = len(str(df[s].max()))
    lmap = {n: 'S'+str(n).zfill(fill) for n in df[s].unique()}
    df[s] = df[s].map(lmap).astype(str)
    df[['start', 'dur']] = df[['start', 'dur']].astype(str)

    df['ch'] = '1'
    df['env'] = 'S'
    df['typ'] = 'U'
    df = df[['ch', 'start', 'dur', 'gen', 'env', 'typ', s]]

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


def make_spk(dfs, path='/home/cilsat/data/speech/rapat', min_dur=12000):
    for n in dfs.cls.unique():
        dfc = dfs.loc[dfs.cls == cls]
        cum = dfc.dur.cumsum()
        dur = cum.max()
        if cum.max() < min_dur: print('not enough data')
        else:
            df = dfc.loc[:cum.loc[cum > min_dur].index[0]].copy()
            # get start and end of segments in samples
            times = np.dstack((df.start.values, (df.start + df.dur).values))
            trims = ['='+str(n)+'s' for n in times.flatten()*160]
            # write segments to file
            old = os.path.join(path, df.file.iloc[0])
            new = 's'+str(cls).zfill(3)
            run(['sox', old, os.path.join(path, 'spk/'+new+'.wav'),
                'trim'] + trims + ['gain', '-6', 'highpass', '120'],
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
            df.file = new
            df.drop(['spkr', 'lbl'], axis=1, inplace=True)
            df.start = np.append([0], df.dur.cumsum()[:-1].values)
            lbl2seg(df, path=False, s='cls').to_csv(os.path.join(path, 'spk/'+new+'.seg'), sep=' ', header=None)
            return df


def make_ubm(out, path='/home/cilsat/data/speech/rapat', min_dur=9000, min_spk=3, max_spkr=120):
    # concat all lbl files and make unique clusters
    dfs = lbl2df(path, 1)
    # get all clusters that are not the first min_spk speakers in each file
    # the assumption here is that the first min_spk speakers are repeated across
    # meetings
    spk = dfs.loc[dfs.groupby(dfs.file).apply(lambda x: x.loc[x.lbl.isin(x.lbl.unique()[min_spk:])]).index.get_level_values(1)]
    spk = spk.loc[spk.lbl > 1]
    # find clusters that have at least min_dur seconds of speech and get them
    segs = []
    count = 0
    for n in spk.cls.unique():
        if count >= max_spkr: break
        dfc = spk.loc[spk.cls == n]
        cum = dfc.dur.cumsum()
        if cum.max() > min_dur:
            df = dfc.loc[:cum.loc[cum > min_dur].index[0]].copy()
            # get start and end of segments in samples
            times = np.dstack((df.start.values, (df.start + df.dur).values))
            trims = ['='+str(n)+'s' for n in times.flatten()*160]
            # write segments to file
            old = os.path.join(path, df.file.iloc[0])
            new = 'c'+str(n).zfill(3)
            run(['sox', old, os.path.join(path, 'ubm/'+new+'.wav'), 'trim'] + trims,
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
            df['bak'] = df.file.str[:-4]
            df.file = new
            df.drop(['spkr', 'lbl'], axis=1, inplace=True)
            segs.append(df)
            count += 1

    segs = pd.concat(segs)
    segs.start = np.append([0], segs.dur.cumsum()[:-1].values)
    infiles = [os.path.join(path, 'ubm/'+n+'.wav') for n in segs.file.unique()]
    cmd = ['sox'] + infiles + [os.path.join(path, 'ubm/'+out+'.wav'), 'gain', '-6', 'highpass', '120']
    run(cmd, stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
    name = os.path.join(path, 'ubm/' + out)
    segs.to_csv(name+'.lbl', sep=' ')
    lbl = lbl2seg(segs, path=False, s='cls')
    lbl.cls = 'S0'
    lbl.to_csv(name+'.seg', sep=' ', header=None)
    return segs


if __name__ == "__main__":
    path = sys.argv[1]
    df = seg2df(path)
    fix_lbl(path, df)
