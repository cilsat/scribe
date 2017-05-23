#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
from scipy.stats import multivariate_normal
from numpy import trace
from numpy.linalg import det
from numpy.linalg import norm
from numpy.linalg import inv

from ..utils.iface import ali2df
from ..feats.feats import align_phones
from ..feats.segmentaxis import segment_axis as sa

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

def calc_bic(x, y, theta=1.82):
    px = np.log(det(np.cov(x, rowvar=False)))
    py = np.log(det(np.cov(x, rowvar=False)))
    pz = np.log(det(np.cov(np.vstack((x, y)), rowvar=False)))
    p = x.shape[1]
    # Nz/2 log|CovZ| - Nx/2 log|CovX| - Nx/2 log|CovY| - lambda*P
    return len(x)*(pz - 0.5*(px + py)) - 0.25*p*(p + 3)*np.log(len(x))*theta

def calc_kl2(x, y):
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cix = inv(cx)
    ciy = inv(cy)
    dxy = np.mean(x, axis=0) - np.mean(y, axis=0)
    return trace((cx - cy)*(ciy - cix)) + trace((ciy + cix)*np.outer(dxy, dxy))

def calc_kl(x, y):
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cix = inv(cx)
    ciy = inv(cy)
    # (mx - my)*(mx - my).T
    dxy = np.outer(np.mean(x, axis=0) - np.mean(y, axis=0))
    return 0.5*trace((cx - cy)*(ciy - cix)) + 0.5*trace((ciy - cix)*dxy)

def calc_per_frame(ali, win_size=150, theta=1.82):
    dim = len(ali.columns) - 4
    pen = 0.25*dim*(dim + 3)*theta
    lbl = ali.loc[ali.turn.diff() != 0].ord

    win_start = 0
    while True:
        win = ali.loc[win_start:win_start + win_size]

def segment3(ali, win_size=500, theta=1.82):
    # get frame dimensions and calculate BIC penalty
    dim = len(ali.columns) - 3
    pen = 0.25*dim*(dim + 3)*theta
    # identify real segment changes
    lbl = ali.loc[ali.turn.diff() != 0].ord

    # main loop: works on frames aligned to segment boundaries
    changes = []
    win_start = 0
    win_last = ali.ord.iloc[-1]
    while True:
        end_frame = ali.loc[ali.ord == win_start].index[0] + win_size
        if win_start >= win_last or end_frame > len(ali): break

        win_end = int(ali.loc[end_frame].ord)
        win = ali.loc[(ali.ord >= win_start) & (ali.ord <= win_end)]
        pz = len(win)*np.log(det(win.drop(
            ['ord','turn','phon'],axis=1).cov()))
        penalty = pen*np.log(2*len(win))

        bic_start = int(ali.loc[win.index[0] + dim].ord + 1)
        bic_end = int(ali.loc[win.index[-1] - dim].ord)
        bics = []
        win_segs = range(bic_start, bic_end)
        for n in win_segs:
            x = ali.loc[(ali.ord >= win_start) & (ali.ord < n)].drop(
                    ['ord', 'turn', 'phon'], axis=1)
            y = ali.loc[(ali.ord >= n) & (ali.ord <= win_end)].drop(
                    ['ord', 'turn', 'phon'], axis=1)
            px = len(x)*np.log(det(x.cov()))
            py = len(y)*np.log(det(y.cov()))
            bics.append(pz - px - py - penalty)

        if len(bics) > 0 and np.max(bics) > 0:
            change = win_segs[np.argmax(bics)]
            print(change, np.max(bics), len(win.turn.unique()) > 1)
            changes.append(change)
            win_start = change
        else:
            win_start = win_end

    return np.array(changes)

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
    vad = ali2df(os.path.join(path, name+'.vad'), 'vad')

    fidx = phon.index.unique()
    fmap = dict(zip(fidx, range(len(fidx))))
    phon.index = phon.index.map(lambda x: fmap[x])
    mfcc.index = mfcc.index.map(lambda x: fmap[x])
    vad.index = vad.index.map(lambda x: fmap[x])

    ali = align_phones(mfcc, phon, seq=True)
    count = ali.groupby([ali.index, ali.ord]).phon.count().values.tolist()
    ali.ord = [l for n in range(len(count)) for l in [n]*count[n]]
    ali['turn'] = np.array(ali.index, dtype=np.uint16)
    ali['vad'] = vad

    if not int_idx:
        rmap = dict(zip(range(len(fidx)), fidx))
        ali.index = ali.index.map(lambda x: rmap[x])

    ali.reset_index(drop=True, inplace=True)
    return ali

def test(name, win=200, theta=1.0):
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    ali = preprocess(name)
    vcd = ali.loc[ali.vad].reset_index(drop=True).drop(['ord', 'phon', 'vad'], axis=1)
    lbl = vcd.loc[vcd.turn.diff() > 0].index
    vcd.drop('turn', axis=1, inplace=True)

    calc = np.array([(n+win, calc_bic(i[:win], i[win:], theta), calc_kl2(i[:win], i[win:])) for n, i in enumerate(sa(vcd.values, win*2, win*2-1, axis=0))])

    plt.plot(calc[:, 0], savgol_filter(calc[:, 1], 101, 3)/calc[:,1].std())
    plt.plot(calc[:, 0], savgol_filter(calc[:, 2], 101, 3)/calc[:,2].std())
    plt.plot(np.arange(len(calc)), np.zeros(len(calc)))
    plt.plot(lbl, [0]*len(lbl), '.')
