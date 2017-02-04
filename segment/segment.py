#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
from scipy.stats import multivariate_normal

from ..utils.iface import ali2df 
from ..feats.feats import align_phones

def calc_glr(df, seg=(0, 0, 0), theta=2.0):
    x = df.loc[(df.ord > seg[0]) & (df.ord < seg[1])].drop(['ord', 'phon'], axis=1)
    y = df.loc[(df.ord > seg[1]) & (df.ord < seg[2])].drop(['ord', 'phon'], axis=1)
    z = pd.concat((x, y))
    xm = multivariate_normal.logpdf(x, x.mean(), x.cov())
    ym = multivariate_normal.logpdf(y, y.mean(), y.cov())
    zm = multivariate_normal.logpdf(z, z.mean(), z.cov())
    return (np.sum(zm) - np.sum(np.hstack((xm, ym))))/len(z)**theta

def sil_segment(df_ali, theta=2.0):
    # silence at the beginning, inside, and ends surrounding speaker segments
    sil_b, sil_i, sil_e = (0, 0, 0)
    ords, glrs = [], []

    phones = df_ali.groupby([df_ali['ord'], df_ali.index]).phon.first()
    for (n, f), p in phones.iteritems():
        if p <= 5 and n - sil_e > 1:
            sil_b, sil_i, sil_e = (sil_i, sil_e, n)
            if sil_b != sil_i != sil_e:
                change = phones.loc[sil_i - 1].index != phones.loc[sil_i + 1].index
                try:
                    glrs.append(calc_glr(df_ali, (sil_b, sil_i, sil_e), theta))
                    ords.append((sil_i, f, change))
                except:
                    print(n, f, p, sil_b, sil_i, sil_e)
            f_prev = f
    return pd.Series(glrs, index=ords)

def preprocess(name, path='.'):
    phon = ali2df(os.path.join(path, name+'.ctm'), 'phon')
    mfcc = ali2df(os.path.join(path, name+'.mfc'), 'delta')
    ali = align_phones(mfcc, phon)
    #p = pd.read_csv('phones.txt', delimiter=' ', header=None, names=['phone', 'num'])
    #ali['phon'] = ali['phon'].map(p.to_dict()['phon'])

    return ali
    
