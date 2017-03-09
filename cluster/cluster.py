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
    ali.ord = [l for n in range(len(count)) for l in [n]*count[n]]

    if int_idx:
        fsmap = dict([(fmap[f], smap[s]) for f in fidx for s in sidx if f[:9] == s])
        ali['spkr'] = ali.index.map(lambda x: fsmap[x])
    else:
        rmap = dict(zip(range(len(fidx)), fidx))
        ali.index = ali.index.map(lambda x: rmap[x])
        ali['spkr'] = ali.index.str[:9]

    ali['turn'] = ali.index
    ali.reset_index(drop=True, inplace=True)

    return ali
