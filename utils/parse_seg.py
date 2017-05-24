#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from subprocess import run, PIPE


def seg2df(path):
    with open(path) as f:
        lines = [n for n, r in enumerate(f.read().splitlines()) if r.startswith(';;')]
    df = pd.read_csv(path, skiprows=lines, delimiter=' ', header=None,
            usecols=[2,3,7], names=['start', 'dur', 'spkr'])
    df.spkr = df.spkr.str[1:].astype(np.int16)
    f = lambda x: timedelta(seconds=x/100.)
    df["start_t"] = df.start.apply(f)
    df["dur_t"] = df.dur.apply(f)
    df[:] = df.sort_values('start').reset_index(drop=True)
    return df


def fix_lbl(path, df):
    # "spkr 0" is assumed to be silence and is assumed to be accurate
    dfspk = df.loc[df.spkr != 0]

    # function to play particular segment
    p = lambda x: run(['play', path, 'trim', str(dfspk.loc[x].start * 0.01),
        str(dfspk.loc[x].dur * 0.01)])
    p2 = lambda x, y: run(['play', path, 'trim', str(dfspk.loc[x].start * 0.01),
        str(dfspk.loc[x:y].dur.sum() * 0.01)])

    # get hypothesized speaker change points
    smap = {n: n for n in dfspk.spkr.unique()}


def play(seg, df):
    for n, i in df.iterrows():
        print(n, i.spkr, i.start, i.dur)
        run(['play', seg, 'trim', str(i.start*0.01), str(i.dur*0.01)])


def splay(seg, df, *args):
    for spk in args:
        play(seg, df.loc[df.spkr == spk])


if __name__ == "__main__":
    path = sys.argv[1]
    df = seg2df(path)
    fix_lbl(path, df)
