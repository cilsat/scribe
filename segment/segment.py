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

def calc_aicc(df, idx, win):
    x = df.iloc[idx-win:idx].drop(['turn', 'ord', 'phon'], axis=1)
    y = df.iloc[idx:idx+win].drop(['turn', 'ord', 'phon'], axis=1)
    px = np.log(det(x.cov()))
    py = np.log(det(y.cov()))
    pz = np.log(det(pd.concat((x, y)).cov()))
    return win*pz - 0.5*win*(px + py) 

def calc_bic(df, idx, win):
    x = df.iloc[idx-win:idx].drop(['turn', 'ord', 'phon'], axis=1)
    y = df.iloc[idx:idx+win].drop(['turn', 'ord', 'phon'], axis=1)
    px = np.log(det(x.cov()))
    py = np.log(det(y.cov()))
    pz = np.log(det(pd.concat((x, y)).cov()))
    return win*pz - 0.5*win*(px + py) 

def segment(df_ali, win=150, theta=1.0):
    d = len(df_ali.columns)
    p = 0.25*d*(d + 3)*np.log(2*win)*theta

    seg = df_ali.reset_index().groupby(df_ali.ord)['index'].first()
    seg = seg.loc[(seg > win) & (seg < len(df_ali) - win)]
    bic = np.array([calc_bic(df_ali, n, win) - p for n in seg])

    return pd.Series(bic, index=seg)

def calc_bic2(x, y, pz, penalty=409.5, theta=1.82):
    px = np.log(np.linalg.det(x.cov()))
    py = np.log(np.linalg.det(y.cov()))
    return pz - len(x)*px - len(y)*py - penalty*theta

def segment3(ali, win_size=500, theta=1.82):
    dim = len(ali.columns) - 3
    pen = 0.25*dim*(dim + 3)*theta
    segs = ali.reset_index().groupby(ali.ord).first()['index']
    find_seg = lambda seg, size: segs[segs > segs.loc[seg] + size].index.min()

    win_start = 0
    while True:
        end_frame = ali.loc[ali.ord == win_start].index[0] + win_size
        if end_frame > len(ali): break

        win_end = int(ali.loc[end_frame].ord + 1)
        win = ali.loc[(ali.ord >= win_start) & (ali.ord < win_end)]
        pz = len(win)*np.log(np.linalg.det(win.drop(['ord','turn','phon'],axis=1).cov()))

        bic_start = int(ali.loc[win.index[0] + dim].ord + 1)
        bic_end = int(ali.loc[win.index[-1] - dim].ord)

        bics = []
        win_segs = range(bic_start, bic_end)
        for n in win_segs:
            x = ali.loc[(ali.ord >= win_start) & (ali.ord < n)].drop(
                    ['ord', 'turn', 'phon'], axis=1)
            y = ali.loc[(ali.ord >= n) & (ali.ord < win_end)].drop(
                    ['ord', 'turn', 'phon'], axis=1)
            px = np.log(np.linalg.det(x.cov()))
            py = np.log(np.linalg.det(y.cov()))
            bics.append(pz - len(x)*px - len(y)*py - pen*np.log(2*len(win)))

        if len(bics) > 0 and np.max(bics) > 0:
            change = np.argmax(bics)
            print(win_segs[change], np.max(bics))
            win_start = win_segs[change] + 1
        else:
            win_start = win_end

        #print(win.index[0], win_start, len(win.turn.unique()) > 1, bics)

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
    ali['turn'] = np.array(ali.index, dtype=np.uint16)
    
    if not int_idx:
        rmap = dict(zip(range(len(fidx)), fidx))
        ali.index = ali.index.map(lambda x: rmap[x])

    ali.reset_index(drop=True, inplace=True)
    return ali

