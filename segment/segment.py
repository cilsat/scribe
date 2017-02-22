#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
from scipy.stats import multivariate_normal
from numpy.linalg import det

from ..utils.iface import ali2df
from ..feats.feats import align_phones

def calc_glr(df, seg=(0, 0, 0), theta=2.0):
    x = df.loc[(df.ord > seg[0]) & (df.ord < seg[1])].drop(['ord', 'phon'], axis=1)
    y = df.loc[(df.ord > seg[1]) & (df.ord < seg[2])].drop(['ord', 'phon'], axis=1)

    """
    if len(x) < len(x.columns) - 1 or len(y) < len(y.columns):
        print(x.iloc[0].name, y.iloc[0].name, "singular")
        return
    """

    z = pd.concat((x, y))
    xm = multivariate_normal.logpdf(x, x.mean(), x.cov())
    ym = multivariate_normal.logpdf(y, y.mean(), y.cov())
    zm = multivariate_normal.logpdf(z, z.mean(), z.cov())
    return (np.sum(zm) - np.sum(np.hstack((xm, ym))))/len(z)**theta

def calc_bic(df, idx, win):
    x = df.iloc[idx-win:idx].drop(['spkr', 'ord', 'phon'], axis=1)
    y = df.iloc[idx:idx+win].drop(['spkr', 'ord', 'phon'], axis=1)
    px = np.log(det(x.cov()))
    py = np.log(det(y.cov()))
    pz = np.log(det(pd.concat((x, y)).cov()))
    return win*pz - 0.5*win*(px + py) 

def segment(df_ali, win=150):
    d = len(df_ali.columns)
    p = 0.25*d*(d + 3)*np.log(2*win)

    seg = ali.reset_index().groupby(ali.ord)['index'].first()

    bic = np.array([calc_bic(df_ali, n, win) - p for n in seg if n > win and n < len(df_ali) - win])

def sil_segment(df_ali, theta=2.0):
    """
    main function for detecting speaker changes given a dataframe containing
    phone-aligned frame information. the dataframe must be sorted by
    """
    # silence at the beginning, inside, and ends surrounding speaker segments
    sil_b, sil_i, sil_e = (0, 0, 0)
    ords, glrs = [], []
    n_len = 0

    dfg = df_ali.groupby([df_ali['ord'], df_ali.index])
    phones = pd.concat((dfg.phon.first(), dfg.ord.count()), axis=1)
    print(phones)
    for (n, f), (p, o) in phones.iterrows():
        # consider only pauses/silence
        if p <= 5:
            # check length of current segment and ensure it's larger than N
            n_len = phones.loc[list(range(sil_e+1, n))].ord.sum()
            print(n_len, sil_e, n)
            if n_len > len(df_ali.columns) - 2:
                sil_b, sil_i, sil_e = (sil_i, sil_e, n)
                if sil_b != sil_i != sil_e:
                    change = phones.loc[sil_i - 1].index != phones.loc[sil_i + 1].index
                    try:
                        glrs.append(calc_glr(df_ali, (sil_b, sil_i, sil_e), theta))
                        ords.append((sil_i, f, change))
                    except:
                        print(n, f, p, sil_b, sil_i, sil_e)
            else:
                sil_b, sil_i, sil_e = (sil_b, sil_i, n)
    return pd.Series(glrs, index=ords)

def preprocess(name, path='.', int_idx=False):
    phon = ali2df(os.path.join(path, name+'.ctm'), 'phon')
    mfcc = ali2df(os.path.join(path, name+'.mfc'), 'delta')
    fidx = phon.index.unique()
    fmap = dict(zip(fidx, range(len(fidx))))
    phon.index = phon.index.map(lambda x: fmap[x])
    mfcc.index = mfcc.index.map(lambda x: fmap[x])
    ali = align_phones(mfcc, phon, seq=True)
    count = ali.groupby([ali.index, ali.ord]).phon.count().values.tolist()
    ali.ord = [l for n in range(len(count)) for l in [n]*count[n]]
    
    if not int_idx:
        rmap = dict(zip(range(len(fidx)), fidx))
        ali.index = ali.index.map(lambda x: rmap[x])

    ali['spkr'] = ali.index
    ali.reset_index(drop=True, inplace=True)
    return ali

