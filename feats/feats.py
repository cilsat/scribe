#!/usr/bin/python3

# methods for feature computation and alignment at the frame, segment, and
# utterance levels.

import pandas as pd
import numpy as np
from itertools import repeat

from .segmentaxis import segment_axis as sa
from ..utils.utils import apply_parallel


# calculate delta and delta-delta features in an utterance dataframe
# delta computation modified from https://github.com/jameslyons/
# python_speech_features/blob/master/python_speech_features/base.py
def roll_delta(dfg, n=2):
    w = np.arange(-n, n + 1)
    d = 1. / np.sum([2 * i * i for i in range(1, n + 1)])

    # uses segment_axis to efficiently generate an overlapping window view of
    # the data and apply the weighting/summing delta function
    def delta(nd):
        pad0 = [nd[0] for _ in range(n)]
        pad1 = [nd[-1] for _ in range(n)]
        padded = np.concatenate((pad0, nd, pad1))
        return np.sum(sa(pad, n * 2 + 1, n * 2, 0).swapaxes(1, 2) * w, axis=-1) * d

    nd_d = delta(dfg.values)
    nd_dd = delta(nd_d)

    return pd.concat((dfg,
                      pd.DataFrame(nd_d, index=dfg.index, dtype=np.float16),
                      pd.DataFrame(nd_dd, index=dfg.index, dtype=np.float16)), axis=1)

# computes delta features in parallel


def compute_deltas(df_mfcc, n=2):
    # group features by utterance and compute deltas/delta-deltas
    mg = df_mfcc.groupby(df_mfcc.index)
    df_deltas = apply_parallel(roll_delta, [(g, n) for _, g in mg])

    # fix column names
    c_ = [c for c in df_mfcc.columns]
    c_d = ['d_' + c for c in df_mfcc.columns]
    c_dd = ['dd_' + c for c in df_mfcc.columns]
    df_deltas.columns = c_ + c_d + c_dd
    return df_deltas


# aligns MFCC frames in one dataframe to phone segments in another dataframe
# creates a new column in the MFCC dataframe indicating phone count, i.e. the
# phone order of a particular frame within an utterance
# TODO the MFCC data doesn't actually need to be passed to this function
# NOTE actually it does because otherwise you'll need to concat it afterwards
def count_phones_per_spkr(dfg_mfcc, dfg_phon):
    spk_frames = []
    for utt, frames in dfg_mfcc.groupby(dfg_mfcc.index):
        seg = dfg_phon.loc[utt]
        idx = [(i, r) for i in range(len(seg))
               for r in repeat(seg.phon.iloc[i], seg.dur.iloc[i])]
        dif = len(frames) - len(idx)
        if dif > 0:
            idx.extend([idx[-1] for _ in range(dif)])
        spk_frames.extend(idx)
    spk_frames = np.array(spk_frames, dtype=np.uint16)
    return dfg_mfcc.assign(ord=spk_frames[:, 0], phon=spk_frames[:, 1])

# align MFCC dataframe to phone dataframe.
# returns frame-level alignments for phone segment number (within utterance),
# phone symbol, and speaker.


def align_phones(df_mfcc, df_phon, spk_group=(), seq=False):
    # use the minimal subset of utt/files contained in both dataframes
    # drop utterances with only 1 phone
    dif = set(df_mfcc.index) ^ set(df_phon.index)
    drop = set(df_phon.loc[df_phon.groupby(
        df_phon.index).dur.count() == 1].index)
    drop |= dif
    df_mfcc.drop(drop, inplace=True)
    df_phon.drop(drop, inplace=True)

    # speaker information should be encoded somehow in the utterance filename
    # the grouper is the pattern needed to extract this information
    # in this case, the speaker can be identified by the first 9 chars in file
    if spk_group:
        start, end = spk_group
        dfg_spk_mfcc = df_mfcc.groupby(df_mfcc.index.str[start: end])
        dfg_spk_phon = df_phon.groupby(df_phon.index.str[start: end])
        args = [(g, dfg_spk_phon.get_group(n)) for n, g in dfg_spk_mfcc]
    else:
        dfg_mfcc = df_mfcc.groupby(df_mfcc.index)
        dfg_phon = df_phon.groupby(df_phon.index)
        args = [(g, dfg_phon.get_group(n)) for n, g in dfg_mfcc]

    if seq:
        ali = pd.concat([count_phones_per_spkr(m, p) for m, p in args])
    else:
        ali = apply_parallel(count_phones_per_spkr, args)
    return ali


def calc_segs(spk):
    dfg = spk.groupby([spk.index, spk.ord])

    dur = pd.Series(dfg.eng.count(), name='dur', dtype=np.int16)
    d_dur = dur.groupby(dur.index.get_level_values(0)).diff()
    dd_dur = d_dur.groupby(d_dur.index.get_level_values(0)).diff()
    d_dur = d_dur.fillna(0).astype(np.int16)
    dd_dur = dd_dur.fillna(0).astype(np.int16)
    d_dur.name = 'd_dur'
    dd_dur.name = 'dd_dur'

    mean = dfg.mean()
    phon = mean.phon
    mean.drop('phon', axis=1, inplace=True)
    d_mean = mean.groupby(mean.index.get_level_values(0)).diff()
    dd_mean = d_mean.groupby(d_mean.index.get_level_values(0)).diff()
    d_mean = d_mean.fillna(0).astype(np.float16)
    dd_mean = dd_mean.fillna(0).astype(np.float16)

    mean.columns = ['avg_' + c for c in mean.columns]
    d_mean.columns = ['d_avg_' + c for c in d_mean.columns]
    dd_mean.columns = ['dd_avg_' + c for c in dd_mean.columns]

    return pd.concat((phon, dur, mean, d_dur, d_mean, dd_dur, dd_mean), axis=1)


# compute segment-level features for utterance classification from a phone
# aligned mfcc dataframe. features include per segment frame averages, variances
# and their deltas.
def compute_seg_feats(df_ali, spk_group=(0, 9)):
    if spk_group:
        start, end = spk_group
        args = [[g] for _, g in df_ali.groupby(df_ali.index.str[start:end])]
    else:
        arts = [[g] for _, g in df_ali.groupby(df_ali.index)]

    return apply_parallel(calc_segs, args)


def compute_utt_feats(df_seg):
    """
    df_seg = df_seg.drop(
        df_seg.columns[df_seg.columns.str.contains('phon')], axis=1)
    dfg_feat = df_seg.groupby(df_seg.index.get_level_values(0))
    df_utt = pd.concat((dfg_feat.mean().astype(np.float16),
                       dfg_feat.var().astype(np.float16)), axis=1)
    cols = df_seg.columns
    df_utt.columns = ['avg_'+c for c in cols] + ['var_'+c for c in cols]
    """
    # drop silence segments at the beginning and end of utterances
    df = df_seg.drop(df_seg.loc[df_seg.phon == 1].index)
    # drop phone column
    df = df.drop('phon', axis=1)
    # group by utterance
    dfg_utt = df.groupby(df.index.get_level_values(0))
    # compute normalized means
    df_utt = dfg_utt.mean() / dfg_utt.std()
    # drop utterances with NaN or Inf
    df_utt = df_utt[np.all(np.isfinite(df_utt), axis=1)]

    return df_utt


def compute_vad(feats, context=5, proportion=0.8):
    # assumes first column of feats is log energy
    log_eng = feats[:, 0]
    # simple energy based cutoff, needs adjustment for real time usage
    cutoff = 5 + 0.5 * log_eng.mean()
    # within a context window centred around the current frame, count the
    # number of frames above the cutoff point
    props = np.array([np.sum(win > cutoff) for win in sa(np.concatenate(
        (np.zeros(context), log_eng, np.zeros(context))), context * 2 + 1, context * 2)])
    # return True (voiced) if the number of frames above the cutoff is above a
    # certain proportion
    return props / context > proportion
