#!/usr/bin/env python

import os
import sys
import pandas as pd
from parse_seg import *
from multiprocessing import Pool, cpu_count
from subprocess import run, PIPE, STDOUT, DEVNULL
from argparse import ArgumentParser


def id_spk(name, data_path, model_path, script_path, exp_path, lium_path):
    trainsh = os.path.join(script_path, 'trainSpkr.sh')
    testsh = os.path.join(script_path, 'speakerID.sh')
    trainlog = os.path.join(exp_path, name + '.train.log')
    testlog = os.path.join(exp_path, name + '.test.log')

    src = os.path.join(data_path, name + '.wav')
    seg = os.path.join(data_path, name + '.seg')
    dest = os.path.join(exp_path, name + '.s.wav')
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
    run(['bash', trainsh, sseg, dest, initgmm, gmm, ubm, lium_path, trainlog],
            stdin=PIPE, stdout=DEVNULL)
    # run speaker identification
    run(['bash', testsh, src, seg, gmm, iseg, ubm, lium_path, testlog],
            stdin=PIPE, stdout=DEVNULL)
    # calculate error rate
    hyp = seg2df(iseg)
    hyp.lbl = hyp.lbl.str.split('#').map(lambda x: int(x[-1][1:]))
    ref.drop(['dest', 'spkr', 'src'], axis=1, inplace=True)
    rspkn = ref.loc[ref.lbl > 0].copy()
    rspkn['hyp'] = hyp.loc[rspkn.index, 'lbl']
    rspkn.to_csv(iseg.replace('.seg', '.lbl'), sep=' ')
    return (rspkn.loc[rspkn.lbl != rspkn.hyp].dur.sum(), rspkn.dur.sum())


def main(data, model, script, exp, lium):
    # read files and sanitize
    names = [w.split('.')[0] for w in os.listdir(data) if w.endswith('.wav')]
    args = [(n, data, model, script, exp, lium) for n in names]
    # build speaker models and do speaker id in parallel
    with Pool(cpu_count()) as pool:
        res = np.array(pool.starmap(id_spk, args))
    print('Error rate: ', np.sum(res[:, 0]) / np.sum(res[:, 1]))

    for n in zip(names, res.tolist(), (res[:,0]/res[:,1]).tolist()):
        print(n, )


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
    main(args.data_path, args.model_path, args.script_path, exp_abspath, args.lium_path)
