#!/usr/bin/python3

from .utils import apply_parallel

import numpy as np
import pandas as pd
from itertools import repeat

# read kaldi phone alignment file and return a dataframe indexed by utt/file
# location and duration are in milliseconds and encoded to unsigned integer
# read Kaldi raw MFCC frames file and return a DataFrame indexed by utt/file
def ali2df(ali_file, raw='phon', fold=None):
    if raw == 'phon':
        df = pd.read_csv(ali_file, delimiter=' ', header=None, index_col=0, usecols=[0, 2, 3, 4], names=['index', 'pos', 'dur', 'phon'])
        df.loc[:, ['pos', 'dur']] = (df.loc[:, ['pos', 'dur']]*100).astype(np.uint16)
        df.phon = df.phon.astype(np.uint8)
        if fold and fold >= 0:
            df = df.assign(fold=fold)
            df.fold = df.fold.astype(np.uint8)

    elif raw == 'mfcc':
        with open(ali_file) as f: raw = f.read().split(']')[:-1]
        raw[:] = [r.strip().splitlines() for r in raw]
        df = pd.DataFrame([fr.strip().split() for r in raw for fr in r[1:]], dtype=np.float16)
        df.index = [n for r in raw for n in [r[0].split('[')[0].strip()]*(len(r) - 1)]
        df.columns = ['eng', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12']

    elif raw == 'delta':
        with open(ali_file) as f: raw = f.read().split(']')[:-1]
        raw[:] = [r.strip().splitlines() for r in raw]
        df = pd.DataFrame([fr.strip().split() for r in raw for fr in r[1:]], dtype=np.float16)
        df.index = [n for r in raw for n in [r[0].split('[')[0].strip()]*(len(r) - 1)]
        df.columns = ['c'+str(n) for n in range(39)]

    elif raw == 'vad':
        with open(ali_file) as f: raw = f.read().splitlines()
        vad = []
        for r in raw:
            k, v = r.split('  ')
            vad.extend([(k, int(n)) for n in v[2:-2].split()])
        return pd.Series([v[1] for v in vad], index=[v[0] for v in vad], dtype=np.bool)

    else:
        with open(ali_file) as f: raw = f.read().split(']')[:-1]
        raw[:] = [r.strip().splitlines() for r in raw]
        df = pd.DataFrame([fr.strip().split() for r in raw for fr in r[1:]], dtype=np.float16)
        df.index = [n for r in raw for n in [r[0].split('[')[0].strip()]*(len(r) - 1)]

    return df

# write dataframe as a group in an HDF file with utt/files as separate datasets
def df2hdf(df, df_name, hdf_file):
    df.to_hdf(hdf_file, df_name)

# read all groups in an HDF file to a dataframe
def hdf2df(hdf_file, df_name):
    return pd.read_hdf(hdf_file, df_name)

def srt2df(srt_file, frame_len=10):
    raw = open(srt_file).read().split('\n\n')[:-1]
    srt = pd.DataFrame.from_dict({int(n[0])-1: [n[1]]+n[3:] for n in [i.split() for i in raw]}, orient='index')
    srt.columns = ['start', 'end', 'spkr']
    f = lambda x: (3600000*int(x[:2]) + 60000*int(x[3:5]) + 1000*int(x[6:8]) + int(x[-3:]))/frame_len
    srt.start = srt.start.map(f).astype(int)
    srt.end = srt.end.map(f).astype(int)
    return srt

def parse_files(mfcc_files=[], phon_files=[]):
    mfcc_files.sort()
    phon_files.sort()
    mfcc_args = list(zip(mfcc_files, range(len(mfcc_files)), repeat('mfcc')))
    phon_args = list(zip(phon_files, range(len(phon_files)), repeat('phon')))

    mfcc = []
    phon = []
    if len(mfcc_files) > 0: mfcc = apply_parallel(ali2df, mfcc_args)
    if len(phon_files) > 0: phon = apply_parallel(ali2df, phon_args)

    return mfcc, phon
