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
    try: df.spkr = df.spkr.str[1:].astype(np.int16)
    except: "cannot convert cluster names to integer"
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
        df['src'] = [n]*len(df) if filemap else [i.replace('lbl', 'wav')]*len(df)
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
            print(i.name, i.lbl, i.src, i.dur)
            run(['play', i.src, 'trim', str(i.start*160)+'s', str(i.dur*160)+'s'] + dsp,
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)
    else:
        print(df.name, df.lbl, df.src, df.dur*160)
        run(['play', df.src, 'trim', str(df.start*160)+'s', str(df.dur*160)+'s'] + dsp,
                stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True)


def play(seg, df):
    if type(df) != pd.core.series.Series:
        for n, i in df.iterrows():
            print(n, i.spkr, i.lbl, i.start*0.01/3600, i.dur*0.01)
            run(['play', seg, 'trim', str(i.start*160)+'s', str(i.dur*160)+'s'] + dsp,
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
    else:
        print(df.name, df.spkr, df.lbl, df.start*0.01/3600, df.dur*0.01)
        run(['play', seg, 'trim', str(df.start*160)+'s', str(df.dur*160)+'s'] + dsp,
                stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)


def lbl2seg(name, path=False, s='cls'):
    # name denotes path to src if path=False, dataframe otherwise
    if path:
        name = name.split('.')[0]
        df = pd.read_csv(path, delimiter=' ', index_col=0)
    else: df = name

    df.index = df.src if 'src' in df.columns else [name]*len(df)
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


# df must have at least start, dur, src, and dest columns
def trim_wav(df, src=None, dest=None):
    if src == None and 'src' not in df.columns:
        print("no source file")
        return
    if dest == None and 'dest' not in df.columns:
        print("no destination file")
        return
    if len(df.src.unique()) > 1 or len(df.dest.unique()) > 1:
        print("too many input/output files")
        return
    src = df.src.iloc[0] if 'src' in df.columns else src
    dest = df.dest.iloc[0] if 'src' in df.columns else src

    times = np.dstack((df.start.values, (df.start + df.dur).values))
    trims = ["="+str(n)+"s" for n in times.flatten()*160]
    cmd = ["sox", src, dest, "trim"] + trims + dsp
    run(cmd, stdout=DEVNULL, stderr=DEVNULL)


def make_spk(dfs, out, col='cls', min_dur=9000):
    spk = []
    for n in dfs[col].unique():
        if n <= 0: continue
        dfc = dfs.loc[dfs[col] == n]
        cum = dfc.dur.cumsum()
        #if cum.max() < min_dur: print('not enough data for ' + str(n))
        else:
            df = dfc.loc[:cum.loc[cum > min_dur].index[0]].copy()
            # get start and end of segments in samples
            spk.append(df)

    # segments need to be in ascending order to concatenate with sox
    spk = pd.concat(spk).sort_index()
    dests = spk.dest.unique()
    for f in spk.src.unique():
        dff = spk.loc[spk.src == f].sort_index()
        trim_wav(dff)
    
    if len(dests) > 1:
        run(['sox'] + dests + [out], stdin=PIPE, stdout=DEVNULL)
    spk.to_csv(out.replace('.wav', '.lbl'), sep=' ')
    spk.start = np.append([0], spk.dur.cumsum()[:-1].values)
    seg = lbl2seg(spk, s=col)
    seg.to_csv(out.replace('.wav', '.seg'), sep=' ', header=None)
    return spk


def make_ubm(out, path='/home/cilsat/data/speech/rapat', min_dur=9000, min_spk=3, max_spkr=120):
    # concat all lbl files and make unique clusters
    dfs = lbl2df(path, 1)
    # get all clusters that are not the first min_spk speakers in each file
    # the assumption here is that the first min_spk speakers are repeated across
    # meetings
    spk = dfs.loc[dfs.groupby(dfs.src).apply(lambda x: x.loc[x.lbl.isin(x.lbl.unique()[min_spk:])]).index.get_level_values(1)]
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
            old = os.path.join(path, df.src.iloc[0])
            new = 'c'+str(n).zfill(3)
            run(['sox', old, os.path.join(path, 'ubm/'+new+'.wav'), 'trim'] + trims,
                    stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
            df['src'] = df.src.str[:-4]
            df['dest'] = new
            df.drop(['spkr', 'lbl'], axis=1, inplace=True)
            segs.append(df)
            count += 1

    segs = pd.concat(segs)
    segs.start = np.append([0], segs.dur.cumsum()[:-1].values)
    infiles = [os.path.join(path, 'ubm/'+n+'.wav') for n in segs.src.unique()]
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
