#!/usr/bin/env python

import os
import sys
import pandas as pd
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from lium_utils import seg2df, lbl2seg, lbl2df, id2df
from identify_spk import test

lium_path = '/home/cilsat/downloads/bin/lium_spkdiarization-8.4.1.jar'
sgmm_path = '/home/cilsat/data/speech/rapat/gmm/120s_all_r2/spk.gmm'
ubm_path = '/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm'
ref_dir = '/home/cilsat/data/speech/rapat/used'
hyp_dir = '/home/cilsat/data/speech/rapat/diarize'

parser = ArgumentParser(description='Script to run LIUM speaker identification\
        on segments diarized using LIUM.')
parser.add_argument('--lium', type=str, default=lium_path)
parser.add_argument('--sgmm', type=str, default=sgmm_path)
parser.add_argument('--ubm', type=str, default=ubm_path)
parser.add_argument('--ref', type=str, default=ref_dir)
parser.add_argument('--hyp', type=str, default=hyp_dir)
parser.add_argument('--exp', type=str, default='120s')
parser.add_argument('--stage', type=int, default=1)
args = parser.parse_args()


def main(ref_dir=args.ref, hyp_dir=args.hyp):
    refs = lbl2df(ref_dir, 1)
    srcs = refs.src.unique()

    names = [os.path.split(s)[-1].split('.')[0] for s in srcs]
    ref_dfs = [refs.loc[(refs.src == s) & (refs.cls > 0)].drop(
        ['spkr', 'lbl', 'gen', 'src'], axis=1) for s in srcs]
    hyp_ins = [os.path.join(hyp_dir, os.path.join(
        n, n + '.spl.3.seg')) for n in names]

    if args.stage <= 1:
        with Pool(cpu_count()) as pool:
            pool.starmap(proc, zip(names, srcs, ref_dfs, hyp_ins))

    if args.stage <= 2:
        score = 0
        total = 0
        for h, ref in zip(hyp_ins, ref_dfs):
            hyp_out = h.replace('.3.', '.' + args.exp + '.')
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
    hyp_out = hyp_in.replace('.3.', '.' + args.exp + '.')
    test(args.lium, hyp_in, src, hyp_out, args.sgmm, args.ubm, name,
         hyp_in.replace('seg', 'log'))


def eval(hyps, ref):
    df = id2df(hyps.replace('.3.', '.' + args.exp + '.'))
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
    main()
