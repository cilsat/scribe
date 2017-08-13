#!/usr/bin/env python

import os
import sys
import pandas as pd
from parse_seg import *
from multiprocessing import Pool, cpu_count
from subprocess import run, PIPE, STDOUT, DEVNULL
from argparse import ArgumentParser


def id_spk(name, data_path, model_path, script_path, exp_path, lium_path):
    testsh = os.path.join(script_path, 'speakerID.sh')
    trainsh = os.path.join(script_path, 'trainSpkr.sh')

    src = os.path.join(data_path, name + '.wav')
    dest = os.path.join(exp_path, name + '.s.wav')
    seg = os.path.join(data_path, name + '.seg')
    sseg = os.path.join(exp_path, name + '.s.seg')

    ubm = os.path.join(model_path, 'ubm.gmm')
    initgmm = os.path.join(exp_path, name + '.s.init.gmm')
    gmm = os.path.join(exp_path, name + '.s.gmm')
    iseg = os.path.join(exp_path, name + '.i.seg')

    # make speaker model
    ref = pd.read_csv(os.path.join(data_path, name + '.lbl'), delimiter=' ',
            index_col=0)
    ref['src'] = src
    ref['dest'] = dest
    make_spk(ref, dest, col='lbl')
    run(['bash', trainsh, sseg, src, initgmm, gmm, ubm, lium_path], stdin=PIPE, stdout=DEVNULL)
    # run speaker identification
    run(['bash', testsh, src, seg, gmm, iseg, ubm, lium_path], stdin=PIPE, stdout=DEVNULL)
    # calculate error rate
    hyp = df2seg(iseg)
    hyp.lbl = hyp.lbl.str.split('#').map(lambda x: int(x[-1][1:]))
    ref['hyp'] = hyp.lbl
    ref.drop(['dest', 'spkr'], axis=1, inplace=True)
    ref.rename({'lbl':'ref'}, inplace=True)
    err = ref.loc[(ref['ref'] > 0) & (ref['ref'] != ref['hyp'])].dur.sum()
    print(name, err/ref.loc[ref['ref'] > 0].dur.sum())
    return ref


def main(data, model, script, exp, lium):
    # build speaker models and do speaker id in parallel
    names = [w.split('.')[0] for w in os.listdir(data) if w.endswith('.wav')]
    args = [(n, data, model, script, exp, lium) for n in names]
    with Pool(cpu_count()) as pool:
        res = pool.starmap(id_spk, args)
    return pd.concat(res)


if __name__ == "__main__":
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'data/speech/rapat')
    model_path = os.path.join(home, 'src/kaldi-offline-transcriber/models')
    script_path = os.path.join(home, 'dev/scribe/lium')
    exp_path = 'exp'
    lium_path = os.path.join(home, 'Downloads/lium_spkdiarization-8.4.1.jar')

    parser = ArgumentParser(description="Train speaker models using a portion \
            of speaker data retrieved from reference files (.lbl), and test on \
            the entire file. Calculate error rate and write to csv.")
    parser.add_argument("--data_path", type=str, default=data_path,
            help="Path to directory containing wavs, segs, and lbls.")
    parser.add_argument("--model_path", type=str, default=model_path,
            help="Path to directory containing LIUM UBM, named ubm.gmm.")
    parser.add_argument("--script_path", type=str, default=script_path,
            help="Path to directory containing scripts for speaker training and\
                diarization.")
    parser.add_argument("--exp_path", type=str, default="exp",
            help="Path to directory storing experiment results, relative to the\
                data_path.")
    parser.add_argument("--lium_path", type=str, default=lium_path,
            help="Path to LIUM jar.")

    args = parser.parse_args()
    exp_abspath = os.path.join(data_path, args.exp_path)
    if not os.path.exists(exp_abspath): os.mkdir(exp_abspath)
    main(args.data_path, args.model_path, args.script_path, exp_abspath, args.lium_path).to_csv(os.path.join(exp_abspath, 'res.csv'), sep=' ')
