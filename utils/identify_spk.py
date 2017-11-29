#!/usr/bin/env python
"""
Used to train and test speaker models from DPD data.
Example usage:
    ./identify_spk.py --data_path $HOME/data/speech/rapat \
            --ubm_path $HOME/src/kaldi-offline-transcriber/models/ubm.gmm \
            --lium_path $HOME/down/prog/lium_spkdiarization-8.4.1.jar
    Trains single speaker model for all speakers obtained from *.lbl and *.wav
    in data_path, tests speaker identification, and scores results.
"""
import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from lium_utils import seg2df, lbl2seg, lbl2df, make_spk
from subprocess import run, PIPE, STDOUT, DEVNULL
from argparse import ArgumentParser, RawTextHelpFormatter


def main():
    # default arguments
    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'data/speech/rapat')
    ubm_path = os.path.join(
        home, 'src/kaldi-offline-transcriber/models/ubm.gmm')
    exp_path = os.path.join(data_path, 'exp')
    lium_path = os.path.join(home, 'Downloads/lium_spkdiarization-8.4.1.jar')
    stage = 0

    parser = ArgumentParser(description="Train speaker models using a portion \
of speaker data retrieved from reference files (.lbl), and test \
the entire file. Calculate error rate and write to csv.",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("--data_path", type=str, default=data_path,
                        help="Path to directory containing wavs, segs, lbls.")
    parser.add_argument("--ubm_path", type=str, default=ubm_path,
                        help="Path LIUM UBM.")
    parser.add_argument("--exp_path", type=str, default="exp",
                        help="Path to directory storing experiment results, \
relative to the data_path.")
    parser.add_argument("--lium_path", type=str, default=lium_path,
                        help="Path to LIUM jar.")
    parser.add_argument("--duration", type=int, default=120,
                        help="Duration of speech in seconds to use for \
speaker model training.")
    parser.add_argument("--stage", type=int, default=stage,
                        help="Start at a particular stage of the \
identification process: \n \
    0: Preprocessing of speaker models.\n \
    1: Training of speaker models.\n \
    2: Testing / identification of speaker segments.\n \
    3: Calculate error rates based on ref and hyp.")

    args = parser.parse_args()
    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)
    res = all(args.data_path, args.ubm_path, args.exp_path,
              args.lium_path, args.duration, args.stage)
    # res = multiple(args.data_path, args.ubm_path, args.exp_path,
    # args.lium_path, args.duration, args.stage)
    print(res)
    # print('all: ', res.err.sum() / res.dur.sum())
    # res.to_csv(os.path.join(args.exp_path, 'res.csv'), sep=' ')


def multiple(data, ubm, exp, lium, dur, stage):
    # read files and sanitize
    names = [w.split('.')[0] for w in os.listdir(data) if w.endswith('.wav')]
    args = [(n, data, ubm, exp, lium, dur, stage) for n in names]
    # build speaker models and do speaker id in parallel
    with Pool(cpu_count()) as pool:
        res = np.array(pool.starmap(id_spk, args))
    df = pd.DataFrame(res, columns=['err', 'dur'], index=names)
    return df


def id_spk(name, data_path, ubm_path, exp_path, lium_path, duration, stage):
    # names and paths
    log = os.path.join(exp_path, name + '.t.log')
    src = os.path.join(data_path, name + '.wav')
    dest = os.path.join(exp_path, name + '.s.wav')
    sseg = os.path.join(exp_path, name + '.s.seg')
    slbl = os.path.join(exp_path, name + '.s.lbl')
    ubm = ubm_path
    initgmm = os.path.join(exp_path, name + '.s.init.gmm')
    gmm = os.path.join(exp_path, name + '.s.gmm')
    iseg = os.path.join(exp_path, name + '.i.seg')
    ilbl = os.path.join(exp_path, name + '.i.lbl')
    tseg = os.path.join(exp_path, name + '.t.seg')

    # obtain timings and audio for speaker models
    if stage < 1:
        ref = pd.read_csv(os.path.join(data_path, name + '.lbl'),
                          delimiter=' ', index_col=0)
        ref['src'] = src
        spk = make_spk(ref, dest, min_dur=duration * 100)
        ref = ref.loc[(ref.lbl > 0) & (
            ~ref.index.isin(spk.index))].reset_index(drop=True)

        spk.to_csv(slbl, sep=' ')
        seg = lbl2seg(spk)
        seg.to_csv(sseg, sep=' ', header=None)
        non = lbl2seg(ref)
        non.to_csv(tseg, sep=' ', header=None)

    # initialize speaker models using ubm
    init_cmd = [
        'java', '-cp', lium_path, 'fr.lium.spkDiarization.programs.MTrainInit',
        '--sInputMask=' + sseg, '--fInputMask=' + src,
        '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
        '--emInitMethod=copy', '--tInputMask=' + ubm,
        '--tOutputMask=' + initgmm, name
    ]

    # train speaker models
    train_cmd = [
        'java', '-cp', lium_path, 'fr.lium.spkDiarization.programs.MTrainMAP',
        '--sInputMask=' + sseg, '--fInputMask=' + src,
        '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
        '--tInputMask=' + initgmm, '--emCtrl=1,5,0.01', '--varCtrl=0.01,10.0',
        '--tOutputMask=' + gmm, name
    ]

    # run speaker identification
    test_cmd = [
        'java', '-cp', lium_path,
        'fr.lium.spkDiarization.programs.Identification',
        '--sInputMask=' + tseg, '--fInputMask=' + src, '--sOutputMask=' + iseg,
        '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
        '--tInputMask=' + gmm, '--sTop=5,' + ubm, '--sSetLabel=add', name
    ]

    if stage < 2:
        with open(log, 'w') as f:
            run(init_cmd, stderr=f)
        with open(log, 'a') as f:
            run(train_cmd, stderr=f)

    if stage < 3:
        with open(log, 'a') as f:
            run(test_cmd, stderr=f)

    # calculate error rate
    if stage < 4:
        hyp = seg2df(iseg)
        hyp.lbl = hyp.lbl.str.split('#').map(lambda x: int(x[-1][1:]))
        ref = seg2df(tseg)
        ref['hyp'] = hyp.lbl
        ref.to_csv(ilbl, sep=' ')
        return (ref.loc[ref.lbl != ref.hyp].dur.sum(), ref.dur.sum())


def all(data, ubm, exp, lium, dur, stage):
    """
    Differs from id_spk in that the speaker model is trained from ALL
    sessions / meetings.
    """
    name = 'spk'
    src = os.path.join(exp, name + '.wav')
    sseg = os.path.join(exp, name + '.s.seg')
    slbl = os.path.join(exp, name + '.s.lbl')
    igmm = os.path.join(exp, name + '.init.gmm')
    gmm = os.path.join(exp, name + '.gmm')
    log = os.path.join(exp, name + '.log')

    # Get speaker model timings/audio from each session.
    # Combine all labels/references into the data required to create a single
    # speaker model with a per speaker minimum duration of training audio.
    # This data includes [exp]/spk.s.{seg,lbl,wav}
    # Output test transcripts to [exp]/*.t.seg.
    if stage < 1:
        dfs = lbl2df(data, 1)
        dfs['dest'] = dfs.src.str.replace(data, exp)
        spk = make_spk(dfs.loc[dfs.lbl > 0], out=src,
                       col='cls', min_dur=dur * 100)
        non = dfs.loc[(dfs.lbl > 0) & (~dfs.index.isin(spk.index))]
        seg = lbl2seg(spk, s='cls')
        seg.to_csv(sseg, sep=' ', header=None)
        spk.to_csv(slbl, sep=' ')
        for n in non.src.unique():
            lbl = non.loc[non.src == n]
            seg = lbl2seg(lbl, s='cls')
            seg.to_csv(lbl.dest.values[0].replace('.wav', '.t.seg'), sep=' ',
                       header=None)

    if stage < 2:
        train(lium, sseg, src, ubm, igmm, gmm, name, log)

    names = [w.split('.')[0] for w in os.listdir(data) if w.endswith('.wav')]
    names.sort()

    if stage < 3:
        # read files and sanitize
        args = [(lium, os.path.join(exp, n + '.t.seg'),
                 os.path.join(data, n + '.wav'),
                 os.path.join(exp, n + '.i.seg'),
                 gmm, ubm, n, os.path.join(exp, n + '.log')) for n in names]
        # build speaker models and do speaker id in parallel
        with Pool(cpu_count()) as pool:
            pool.starmap(test, args)

    if stage < 4:
        res = []
        for n in names:
            iseg = os.path.join(exp, n + '.i.seg')
            ilbl = os.path.join(exp, n + '.i.lbl')
            tseg = os.path.join(exp, n + '.t.seg')
            hyp = seg2df(iseg)
            hyp.lbl = hyp.lbl.str.split('#').map(lambda x: int(x[-1][1:]))
            ref = seg2df(tseg)
            ref['hyp'] = hyp.lbl
            ref.to_csv(ilbl, sep=' ')
            ref.index = [n] * len(ref)
            res.append(ref)
        res = pd.concat(res)
        res.drop(['spkr', 'gen'], axis=1, inplace=True)
        res.to_csv(os.path.join(exp, 'results.csv'))
        return res


def train(lium, seg, wav, ubm, igmm, gmm, name, log):
    # initialize speaker models using ubm
    init_cmd = [
        'java', '-cp', lium, 'fr.lium.spkDiarization.programs.MTrainInit',
        '--sInputMask=' + seg, '--fInputMask=' + wav,
        '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
        '--emInitMethod=copy', '--tInputMask=' + ubm, '--tOutputMask=' + igmm,
        name
    ]

    # train speaker models
    train_cmd = [
        'java', '-cp', lium, 'fr.lium.spkDiarization.programs.MTrainMAP',
        '--sInputMask=' + seg, '--fInputMask=' + wav,
        '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
        '--tInputMask=' + igmm, '--emCtrl=1,5,0.01', '--varCtrl=0.01,10.0',
        '--tOutputMask=' + gmm, name
    ]

    with open(log, 'w') as f:
        run(init_cmd, stderr=f)
    with open(log, 'a') as f:
        run(train_cmd, stderr=f)


def test(lium, seg, wav, iseg, gmm, ubm, name, log):
    # identify speaker segments
    cmd = [
        'java', '-cp', lium, 'fr.lium.spkDiarization.programs.Identification',
        '--sInputMask=' + seg, '--fInputMask=' + wav, '--sOutputMask=' + iseg,
        '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
        '--tInputMask=' + gmm, '--sTop=5,' + ubm, '--sSetLabel=add', name
    ]

    with open(log, 'a') as f:
        run(cmd, stderr=f)


if __name__ == "__main__":
    main()
