#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
from scipy.stats import multivariate_normal
from numpy import trace
from numpy.linalg import det
from numpy.linalg import norm
from numpy.linalg import inv

from sklearn.mixture import GaussianMixture as gmm

from ..utils.iface import ali2df
from ..feats.feats import align_phones
from ..feats.segmentaxis import segment_axis as sa

def cusum(z):
    means_init = np.vstack((np.mean(z[:10], axis=0), np.mean(z[-10:], axis=0)))
    clf = gmm(n_components=2, means_init=means_init)
    clf.fit(z)
    c = np.uint16(0.5*z.shape[0])
    return np.sum(mn.logpdf(z[:c], clf.means_[0], clf.covariances_[0])) - np.sum(mn.logpdf(z[c:], clf.means_[1], clf.covariances_[1]))

def glr(x, y, theta=1.82):
    xm = mn.logpdf(x, np.mean(x, axis=0), np.cov(x, rowvar=False))
    ym = mn.logpdf(y, np.mean(y, axis=0), np.cov(y, rowvar=False))
    z = np.vstack((x, y))
    zm = mn.logpdf(z, np.mean(z, axis=0), np.cov(z, rowvar=False))
    return (np.sum(zm) - np.sum(np.hstack((xm, ym))))/len(z)**theta

def bic(x, y, theta=1.82):
    px = np.log(det(np.cov(x, rowvar=False)))
    py = np.log(det(np.cov(x, rowvar=False)))
    pz = np.log(det(np.cov(np.vstack((x, y)), rowvar=False)))
    p = x.shape[1]
    # Nz/2 log|CovZ| - Nx/2 log|CovX| - Nx/2 log|CovY| - lambda*P
    return len(x)*(pz - 0.5*(px + py)) - 0.25*p*(p + 3)*np.log(len(x))*theta

def xbic(x, y):
    px = np.log(det(np.cov(x, rowvar=False)))
    py = np.log(det(np.cov(x, rowvar=False)))
    pz = np.log(det(np.cov(np.vstack((x, y)), rowvar=False)))
    return pz

def kl2(x, y):
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cix = inv(cx)
    ciy = inv(cy)
    dxy = np.mean(x, axis=0) - np.mean(y, axis=0)
    return np.trace((cx - cy)*(ciy - cix)) + np.trace((ciy + cix)*np.outer(dxy, dxy))

def kl(x, y):
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cix = inv(cx)
    ciy = inv(cy)
    # (mx - my)*(mx - my).T
    dxy = np.mean(x, axis=0) - np.mean(y, axis=0)
    ddxy = np.outer(dxy, dxy)
    return 0.5*np.trace((cx - cy)*(ciy - cix)) + 0.5*np.trace(ddxy*(ciy - cix))

def dsd(x, y):
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cix = inv(cx)
    ciy = inv(cy)
    return np.trace((cx - cy)*(ciy - cix))

def gish(x, y):
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    alpha = len(x)/(len(x) + len(y))
    beta = len(y)/(len(y) + len(x))
    w = alpha*cx + beta*cy
    return -0.5*len(x)*np.log(det(cx)**alpha*det(cy)**(1-alpha)/det(w))

def calc_per_frame(ali, win_size=150, theta=1.82):
    dim = len(ali.columns) - 4
    pen = 0.25*dim*(dim + 3)*theta
    lbl = ali.loc[ali.turn.diff() != 0].ord

    win_start = 0
    while True:
        win = ali.loc[win_start:win_start + win_size]

def segment(df, fn=kl2, win=250, thr=2.0):
    f = 0
    l = f + win
    m = np.uint8(0.5*win)
    c = f + m
    hyp = []
    val = []
    while True:
        if l > len(df): break
        if fn(df.iloc[f:f+c].values, df.iloc[f+c:l].values) > thr:
            print("something")
            f = c
        else:
            f = l
        l = f + win
        c = f + m

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


def test_lbl(name, fn=dsd, thr=2.0, win=200, theta=1.83):
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    ali = preprocess(name)
    vcd = ali.loc[ali.vad].reset_index(drop=True).drop(['ord', 'phon', 'vad'], axis=1)
    lbl = vcd.loc[vcd.turn.diff() > 0].index
    raw = vcd.drop('turn', axis=1).values

    calc = np.array([(n+win, fn(i[:win], i[win:])) for n, i in enumerate(sa(raw, win*2, win*2-1, axis=0))])

    plt.plot(calc[:, 0], savgol_filter(calc[:, 1], 101, 3)/calc[:,1].std())
    plt.plot(np.arange(len(calc)), [thr]*len(calc))
    plt.plot(lbl, [thr]*len(lbl), '.')


def test_unk(path, fn=kl2, thr=2.0, win=200, theta=1.83):
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    ali = pd.concat((ali2df(os.path.join(path, 'mfc'), 'delta'),
                     ali2df(os.path.join(path, 'vad'), 'vad')),
                     axis=1)

    vcd = ali.loc[ali.vad].reset_index(drop=True)
    raw = vcd.drop('vad', axis=1).values

    calc = np.array([(n+win, fn(i[:win], i[win:])) for n, i in enumerate(sa(raw, win*2, win*2-1, axis=0))])

    plt.plot(calc[:, 0], savgol_filter(calc[:, 1], 101, 3)/calc[:,1].std())
    plt.plot(np.arange(len(calc)), [thr]*len(calc))


def seg_lbl(wav_path, txt_path, name, fn=kl2, thr=2.0, win=200, theta=1.83):
    from scipy.signal import savgol_filter

    scp, mfc = gen_rand(wav_path, txt_path, name)
    vad = compute_vad(mfc)
    idx = [r for n in scp.index for r in [n]*scp.loc[n, 'length']]
    ali = pd.concat((pd.DataFrame(mfc), pd.Series(vad, name='vad'), pd.Series(idx, name='uttid')), axis=1)
