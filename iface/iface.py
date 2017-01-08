#!/usr/bin/python3

import numpy as np
import pandas as pd
from utils import apply_parallel

# read kaldi phone alignment file and return a dataframe indexed by utt/file
# location and duration are in milliseconds and encoded to unsigned integer
# read Kaldi raw MFCC frames file and return a DataFrame indexed by utt/file
def ali2df(ali_file, raw):
    if raw == 'phon':
        df = pd.read_csv(ali_file, delimiter=' ', header=None, index_col=0, usecols=[0, 2, 3, 4], names=['index', 'pos', 'dur', 'phon'])
        df.loc[:, ['pos', 'dur']] = (df.loc[:, ['pos', 'dur']]*100).astype(np.uint16)
        df['phon'] = df['phon'].astype(np.uint8)

    elif raw == 'mfcc':
        with open(ali_file) as f: raw = f.read().split(']')[:-1]
        raw[:] = [r.strip().splitlines() for r in raw]
        df = pd.DataFrame([fr.strip().split() for r in raw for fr in r[1:]], dtype=np.float16)
        df.index = [n for r in raw for n in [r[0].split('[')[0].strip()]*(len(r) - 1)]
        df.columns = ['eng', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12']

    return df

# write dataframe as a group in an HDF file with utt/files as separate datasets
def df2hdf(df, df_name, hdf_file):
    df.to_hdf(hdf_file, df_name)

# read all groups in an HDF file to a dataframe
def hdf2df(hdf_file, df_name):
    return pd.read_hdf(hdf_file, df_name)

def parse_files(mfcc_files=[], phon_files=[]):
    mfcc_args = [(f, 'mfcc') for f in mfcc_files]
    phon_args = [(f, 'phon') for f in phon_files]

    mfcc = apply_parallel(ali2df, mfcc_args)
    phon = apply_parallel(ali2df, phon_args)

    return mfcc, phon
