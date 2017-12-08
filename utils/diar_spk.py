#!/usr/bin/env python

import os
import sys
import pandas as pd
from multiprocessing import Pool, cpu_count
from lium_utils import seg2df, lbl2seg, lbl2df, id2df
from identify_spk import test


def main(ref_dir, hyp_dir):
    refs = lbl2df(ref_dir, 1)
    srcs = refs.src.unique()

    names = [os.path.split(s)[-1].split('.')[0] for s in srcs]
    ref_dfs = [refs.loc[(refs.src == s) & (refs.cls > 0)].drop(
        ['spkr', 'lbl', 'gen', 'src'], axis=1) for s in srcs]
    hyp_ins = [os.path.join(hyp_dir, os.path.join(
        n, n + '.spl.3.seg')) for n in names]

    # with Pool(cpu_count()) as pool:
    # pool.starmap(proc, zip(names, srcs, ref_dfs, hyp_ins))

    score = 0
    total = 0
    for h, ref in zip(hyp_ins, ref_dfs):
        hyp_out = h.replace('.3.', '.out.')
        hyp = id2df(hyp_out)
        join = pd.concat(
            (hyp.set_index('start'), ref.set_index('start')), axis=1).dropna()
        score += join.loc[join.cls == join.lbl, 'dur'].sum()
        total += join.dur.sum()

    print(score / total)

    # with Pool(cpu_count()) as pool:
    # res = pool.starmap(eval, zip(hyp_ins, ref_dfs))

    # df = pd.concat(res)
    # spk = df.loc[df.ref > 0]
    # print(spk)
    # print(spk.loc[spk.ref == spk.hyp, 'dur'].sum() / spk.dur.sum())


def proc(name, src, ref_df, hyp_in):
    hyp_out = hyp_in.replace('.3.', '.out.')
    test('/home/cilsat/down/prog/lium_spkdiarization-8.4.1.jar', hyp_in, src, hyp_out, '/home/cilsat/data/speech/rapat/gmm/150s_all_r1/spk.gmm',
         '/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm', name, hyp_in.replace('seg', 'log'))


def eval(hyps, ref):
    df = id2df(hyps.replace('.3.', '.out.'))
    scores = []
    refs = []

    for _, hyp in df.iterrows():
        try:
            begin = ref.loc[ref.start <= hyp.start].iloc[-1]
        except Exception as e:
            print(hyp.name, e)
            begin = ref.iloc[0]
        try:
            end = ref.loc[ref.start + ref.dur >= hyp.start + hyp.dur].iloc[0]
        except Exception as e:
            print(hyp.name, e)
            end = ref.iloc[-1]

        score = 0
        if begin.name == end.name:
            if begin.cls == hyp.lbl:
                score += hyp.dur
            refs.append(begin.cls)
        else:
            block = ref.loc[begin.name: end.name]
            dur = block.groupby(block.cls).dur.sum()
            cls = dur.idxmax()
            score += dur.loc[cls]
            refs.append(cls)
        scores.append(score)

    df['score'] = scores
    df['ref'] = refs
    df.score = df.score.astype(int)
    df.ref = df.ref.astype(int)
    return df


if __name__ == "__main__":
    main('/home/cilsat/data/speech/rapat',
         '/home/cilsat/data/speech/rapat/diar')
