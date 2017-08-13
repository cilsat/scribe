#!/usr/bin/env python

import os
import sys
from .utils import *
from multiprocessing import Pool, cpu_count
from subprocess import run, PIPE, STDOUT, DEVNULL

data_path = '/home/cilsat/data/speech/rapat'
model_path = '/home/cilsat/src/kaldi-offline-transcriber/models'
script_path = '/home/cilsat/dev/scribe/lium'


def id_spk(name):
    testsh = os.path.join(script_path, 'speakerID.sh')
    trainsh = os.path.join(script_path, 'trainSpkr.sh')

    src = os.path.join(data_path, name + '.wav')
    dest = os.path.join(data_path, 'exp', name + '.s.wav')
    sseg = os.path.join(data_path, 'exp', name + '.s.seg')

    ubm = os.path.join(model_path, 'ubm.gmm')
    initgmm = os.path.join(data_path, 'exp', name + '.s.init.gmm')
    gmm = os.path.join(data_path, 'exp', name + '.s.gmm')
    iseg = os.path.join(data_path, 'exp', name + '.i.seg')

    # make speaker model
    lbl = pd.read_csv(os.path.join(data_path, name + '.lbl'), delimiter=' ',
            index_col=0)
    lbl['src'] = src
    lbl['dest'] = dest
    make_spk(lbl, dest, col='lbl')
    run(['bash', modelSpkr, seg, wav, initgmm, gmm, ubm], stdin=PIPE, stdout=DEVNULL)
    # run speaker identification
    run(['bash', idSpkr, wav, seg, gmm, iseg, ubm], stdin=PIPE, stdout=DEVNULL)


def main(data, model, script):
    # build speaker models and do speaker id in parallel
    names = [w.split('.')[0] for w in os.listdir(data) if w.endswith('.wav')]
    args = [(n, data, model, script) for n in names]
    with Pool(cpu_count()) as pool:
        res = pool.starmap(id_spk, args)


if __name__ == "__main__":
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    script_path = sys.argv[3]
    main(data_path, model_path, script_path)
