#!/usr/bin/python2

import numpy as np
import pandas as pd
import multiprocessing as mp

# read kaldi phone alignment file and return a dataframe indexed by utt/file
# location and duration are in milliseconds and encoded to unsigned integer
# read Kaldi raw MFCC frames file and return a DataFrame indexed by utt/file
def ali2df(ali_file, raw='phon'):
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
    import h5py
    with h5py.File(hdf_file, 'w') as f:
        g = f.create_group(df_name)
        for i in df.index.unique():
            data = df.loc[i].values
            g.create_dataset(name=i, data=data)
    #df.to_hdf(hdf_file, df_name, mode='a', format='table', data_columns=True)

# read all groups in an HDF file to a dataframe
def hdf2df(hdf_file, group_name):
    import h5py
    with h5py.File(hdf_file) as f: return f[group_name]['table'][:]
