import numpy as np
import pandas as pd
import os

from ..utils.iface import ali2df
from ..feats.feats import align_phones

def preprocess(name, path, int_idx=True):
    phon = ali2df(os.path.join(path, name+'.ctm'), 'phon')
    mfcc = ali2df(os.path.join(path, name+'.mfc'), 'delta')

    # this works with files where speaker ids are encoded by the first 9 letters
    # of the filename
    fidx = phon.index.unique()
    fmap = dict(zip(fidx, range(len(fidx))))
    sidx = fidx.str[:9].unique()
    smap = dict(zip(sidx, range(len(sidx))))

    # map utterance names to int and align phones
    phon.index = phon.index.map(lambda x: fmap[x])
    mfcc.index = mfcc.index.map(lambda x: fmap[x])
    ali = align_phones(mfcc, phon, seq=True)
    count = ali.groupby([ali.index, ali.ord]).phon.count().values.tolist()
    ali.ord = np.array([l for n in range(len(count)) for l in [n]*count[n]], dtype=np.uint16)
    ali['turn'] = np.array(ali.index, dtype=np.uint16)

    if int_idx:
        fsmap = dict([(fmap[f], smap[s]) for f in fidx for s in sidx if f[:9] == s])
        ali['spkr'] = ali.index.map(lambda x: fsmap[x]).astype(np.uint16)
    else:
        rmap = dict(zip(range(len(fidx)), fidx))
        ali.index = ali.index.map(lambda x: rmap[x])
        ali['spkr'] = ali.index.str[:9].astype(np.uint16)

    ali.reset_index(drop=True, inplace=True)
    return ali

# penalty is calculated as 0.25*N*(N + 3) where N is #dimensions
# theta is obtained empirically from f1 score of training data
def calc_bic(x, y, penalty=409.5, theta=1.82):
    px = np.log(np.linalg.det(x.cov()))
    py = np.log(np.linalg.det(y.cov()))
    pz = np.log(np.linalg.det(pd.concat((x, y)).cov()))
    len_z = len(x) + len(y)
    return len_z*pz - len(x)*px - len(y)*py - penalty*np.log(2*len_z)*theta

def cluster_seq(ali):
    for n in ali.turn.unique():
        t = ali.loc[ali.turn == n].drop(['turn'], axis=1)
