#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from subprocess import run, PIPE, DEVNULL


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


def lbl2df(path, start=10):
    lbls = [n for n in os.listdir(path) if n.endswith('.lbl')]
    lbls.sort()
    dfs = []
    cls = start
    for n, i in enumerate(lbls):
        df = pd.read_csv(i, delimiter=' ', index_col=0)
        df['file'] = [n]*len(df)
        cmap = {n: i + cls for i, n in enumerate(df.loc[df.lbl > 0, 'lbl'].unique())}
        for n in range(-2, 1): cmap[n] = n
        df['cls'] = df.lbl.map(cmap)
        cls += len(cmap)
        dfs.append(df)
    dfs = pd.concat(dfs)
    return {n: i for n, i in enumerate(lbls)}, dfs


def cplay(dfs, nfile, lbls, lbl):
    play(lbls[nfile].split('.')[0] + '.mp3', dfs.loc[(dfs.file == nfile) & (dfs.lbl == lbl)])


def play(seg, df):
    for n, i in df.iterrows():
        print(n, i.spkr, i.start*0.01/3600, i.dur*0.01)
        run(['play', seg, 'trim', str(i.start*0.01), str(i.dur*0.01)], stdout=DEVNULL)


def splay(seg, df, *args):
    for spk in args:
        play(seg, df.loc[df.spkr == spk])


def lbl2seg(path):
    lbl = pd.read_csv(path, delimiter=' ', index_col=0)
    cls = [lbl.loc[lbl.lbl == n] for n in lbl.lbl.unique()]


if __name__ == "__main__":
    path = sys.argv[1]
    df = seg2df(path)
    fix_lbl(path, df)
