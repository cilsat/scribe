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


def fix_lbl(name, smap, gmap):
    df = seg2df(seg)


def play(seg, df):
    for n, i in df.iterrows():
        print(n, i.spkr, i.start*0.01/3600, i.dur*0.01)
        run(['play', seg, 'trim', str(i.start*0.01), str(i.dur*0.01)], stdout=DEVNULL)


def splay(seg, df, *args):
    for spk in args:
        play(seg, df.loc[df.spkr == spk])


if __name__ == "__main__":
    path = sys.argv[1]
    df = seg2df(path)
    fix_lbl(path, df)
