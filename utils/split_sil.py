#!/usr/bin/env python3
"""
Split input stream into files on silence.
Example execution:
    ./split_sil.py -s 650 -e 15 -i ~/data/speech/rapat/m0002-0.wav \
            -o ~/dev/scribe/test/splits

Don't forget to use the correct UBM!
Use the speaker model obtained from 120s_all
"""

import os
import sys
import argparse
import tempfile
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import pandas as pd
from threading import Thread
from multiprocessing import Pool, cpu_count
from tempfile import mkstemp
from identify_spk import test
from lium_utils import lbl2df, cplay
from segmentaxis import segment_axis


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-t', '--stream', action='store_true',
    help='type of input, either stream or file')
parser.add_argument(
    '-r', '--samplerate', type=int, default=44100, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-b', '--blocksize', type=int, default=1024,
    help='block size of stream')
parser.add_argument(
    '-s', '--split-thr', type=float, default=350,
    help='length of silence (in ms) before the signal is split')
parser.add_argument(
    '-e', '--energy-thr', type=float, default=-30,
    help='cutoff energy for silence in -db')
parser.add_argument(
    '-i', '--in-file', type=str, help='path to input file')
parser.add_argument(
    '-o', '--out-dir', type=str, help='path to directory to store splits')
parser.add_argument(
    '--stage', type=int, default=0,
    help='which stage of the experiment to begin on')
parser.add_argument(
    '--data-path', type=str, help='Path to labels and wavs')
parser.add_argument(
    '--sm', type=str, help='Path to speaker model')
parser.add_argument(
    '--ubm', type=str, help='Path to universal background model')
parser.add_argument(
    '--lium', type=str, help='Path to LIUM jar')
parser.add_argument(
    '--ncpu', type=int, default=cpu_count(),
    help='Number of processing cores to utilize for experiment')
parser.add_argument(
    '--win-size', type=int, default=3,
    help='Window size for maximum average window method')
args = parser.parse_args()


def main():
    print(args)
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    # if args.stream:
        # stream_input()
    # else:
        # file_input()

    names = [n.split('.')[0] for n in os.listdir(args.data_path)
             if n.endswith('.lbl')]

    # names = ['m0057-0']

    paths = {n: os.path.join(args.out_dir, n) for n in names}

    if args.stage < 2:
        for n in names:
            if not os.path.exists(paths[n]):
                os.mkdir(paths[n])
            file_input(in_file=os.path.join(
                args.data_path, n + '.wav'), out_dir=paths[n])

    if args.stage < 3:
        map_args = [(n, paths[n]) for n in names]
        with Pool(args.ncpu) as p:
            p.starmap(lium_test, map_args)

    if args.stage < 4:
        map_args = [(n, paths[n], args.data_path) for n in names]
        with Pool(cpu_count()) as p:
            p.starmap(lium_score, map_args)
        df_all = pd.concat([pd.read_csv(os.path.join(
            paths[n], n + '_info.csv'), index_col=0) for n in names])
        df_all.to_csv(os.path.join(args.out_dir, 'results.csv'))

    if args.stage > 3:
        maw_args = [(n, paths[n], args.win_size) for n in names]
        with Pool(cpu_count()) as p:
            dfs = p.starmap(lium_maw, maw_args)
        df_maw = pd.concat(dfs)
        df_maw.to_csv(os.path.join(
            args.out_dir, str(args.win_size) + '-maw.csv'))


def file_input(split_thr=args.split_thr, energy_thr=args.energy_thr,
               in_file=args.in_file, out_dir=args.out_dir,
               blocksize=args.blocksize):
    info = sf.info(in_file)
    base = os.path.basename(in_file).split('.')[0]
    fill = int(np.log10(info.duration * 100)) + 1
    samplerate = info.samplerate
    mult = 100 / samplerate

    sil_sum = 0
    # number of silent blocks to tolerate before splitting
    split_thr = int(info.samplerate * split_thr / (1000 * blocksize))
    # energy cutoff
    energy_thr = 10**(0.1 * energy_thr)

    buf = []
    starts = []
    durs = []
    names = []
    for n, block in enumerate(sf.blocks(in_file, blocksize=blocksize)):
        rms = np.sqrt(np.mean(block**2))
        if rms < energy_thr:
            sil_sum += 1
            if not buf:
                continue
            else:
                buf.extend(block)

            if sil_sum > split_thr:
                dur = len(buf) * mult
                durs.append(dur)
                start = n * blocksize * mult - dur
                starts.append(start)
                name = base + '_' + str(int(start)).zfill(fill) + '.wav'
                names.append(name)
                sf.write(os.path.join(out_dir, name), buf,
                         samplerate=samplerate, subtype=info.subtype,
                         format=info.format)
                with open(os.path.join(out_dir, name.replace(
                        '.wav', '.uem.seg')), 'w') as f:
                    f.write(name + ' 1 0 ' + str(int(dur)) + ' U U U S1')
                buf = []
        else:
            sil_sum = 0
            buf.extend(block)

    df = pd.DataFrame(index=names, data={'start': starts, 'dur': durs})
    df.start = df.start.round().astype(int)
    df.dur = df.dur.round().astype(int)
    df.to_csv(os.path.join(out_dir, base + '_info.csv'))


def stream_input():
    block_q = queue.Queue()

    sil = True
    sil_sum = 0
    # number of silent blocks to tolerate before splitting
    split_thr = int(args.samplerate * args.split_thr / (1000 * args.blocksize))
    # energy cutoff
    energy_thr = 10**(0.1 * args.energy_thr)

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        block_q.put(indata.copy())

    buf = []
    try:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels, blocksize=args.blocksize,
                            callback=callback):
            while True:
                block = block_q.get()
                rms = np.sqrt(np.mean(block**2))

                if rms < energy_thr:
                    sil_sum += 1
                    if not buf:
                        continue
                    else:
                        buf.extend(block)

                    if sil_sum > split_thr:
                        fd, name = mkstemp(suffix='.wav', dir=args.out_dir)
                        sf.write(name, buf, samplerate=args.samplerate,
                                 subtype='PCM_16')
                        print(name, 100 * len(buf) / args.samplerate)
                        buf = []
                else:
                    sil_sum = 0
                    buf.extend(block)

    except KeyboardInterrupt:
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))


def lium_test(name, out_dir):
    info = os.path.join(out_dir, name + '_info.csv')
    df = pd.read_csv(info, index_col=0)

    # Use LIUM to identify speakers
    for n in df.index:
        seg = os.path.join(out_dir, n.replace('.wav', '.uem.seg'))
        wav = os.path.join(out_dir, n)
        log = os.path.join(out_dir, n.replace('.wav', '.log'))
        iseg = seg.replace('.uem.seg', '.i.seg')
        lbl = n.split('.')[0]
        test(lium=args.lium, seg=seg, wav=wav, iseg=iseg,
             gmm=args.sm, ubm=args.ubm, name=lbl, log=log)

    # Get results of speaker identification and stuff them into info
    hyp = []
    for n in df.index:
        with open(os.path.join(out_dir, n.replace('.wav', '.i.seg'))) as f:
            hyp.append(int(f.read().split('#')[-1][1:-1]))

    df['hyp'] = hyp
    df.to_csv(info)


def lium_score(name, out_dir, out_res=False, hyp_col='hyp',
               data_dir='/home/cilsat/data/speech/rapat'):
    """
    Align output of lium_test with reference files and score accordingly
    """
    dfs = lbl2df(data_dir, 1)
    src = os.path.join(os.path.dirname(dfs.src.iloc[0]), name + '.wav')
    ref = dfs.loc[dfs.src == src]
    if len(ref) == 0:
        print('source not found')
        sys.exit(0)

    info = os.path.join(out_dir, name + '_info.csv')
    df = pd.read_csv(info, index_col=0)

    scores = []
    refs = []
    for key, hyp in df.iterrows():
        # get last ref segment that starts before hyp starts
        try:
            begin = ref.loc[ref.start <= hyp.start].iloc[-1]
        except Exception as e:
            print(hyp.name, e)
            begin = ref.iloc[0]
        # get first ref segment that ends after hyp ends
        try:
            end = ref.loc[ref.start + ref.dur >= hyp.start + hyp.dur].iloc[0]
        except Exception as e:
            print(hyp.name, e)
            end = ref.iloc[-1]
        # calculate how many frames hyp identifies correctly
        score = 0
        # if all of hyp is contained within one ref segment
        if begin.name == end.name:
            if begin.cls == hyp[hyp_col]:
                score += hyp.dur
        else:
            if begin.cls == hyp[hyp_col]:
                score += begin.start + begin.dur - hyp.start
                refs.append(begin.cls)
            if end.cls == hyp[hyp_col]:
                score += hyp.start + hyp.dur - end.start
                refs.append(end.cls)
            for n in range(begin.name + 1, end.name):
                if ref.loc[n, 'cls'] == hyp.hyp:
                    score += ref.loc[n, 'dur']
        scores.append(score)
        refs.append(begin.cls)

    df['score'] = scores
    df['ref'] = refs
    df.score = df.score.astype(int)
    df.ref = df.ref.astype(int)
    df.to_csv(info)

    if out_res:
        return df


def lium_parse_log(log):
    with open(log) as f:
        raw = f.read().split('Picked up _JAVA_OPTIONS')[-1].splitlines()[45:]

    # hyp = {int(x[5][1:-1]): float(x[6])
        # for x in (r.split()
        # for r in open(log).readlines()[45:]
        # if r.find('score S') > 0)}

    hyp = []
    for r in raw:
        b = r.find('score S')
        if b > 0:
            x = r.split()
            hyp.append(float(x[6]))

    return np.array(hyp)


def lium_maw(name, out_dir, win_size=3):
    df = pd.read_csv(os.path.join(out_dir, name + '_info.csv'), index_col=0)
    logs = [os.path.join(out_dir, l.replace('wav', 'log')) for l in df.index]

    # Determine speakers from testing log files
    with open(logs[0]) as f:
        raw = f.read().split('Picked up _JAVA_OPTIONS')[-1].splitlines()[45:]
    spkr = [int(x[5][1:-1])
            for x in (r.split()
                      for r in raw if r.find('score S') > 0)]

    # Parse logs to obtain id scores for all speakers
    spk_scores = np.vstack([lium_parse_log(l) for l in logs])
    dfspk = pd.concat(
        (df, pd.DataFrame(
            spk_scores,
            columns=['S' + str(s) for s in spkr], index=df.index)), axis=1)
    dfspk.to_csv(os.path.join(out_dir, name + '_full.csv'))

    # Pad beginning and end of parsed logs
    pad = np.zeros(((win_size - 1), len(spkr)))
    dur = 0.1 * df.dur.values.reshape((len(df.dur), 1))
    nhyp = np.vstack((pad, spk_scores * dur))

    # Get best hypothesis from an averaged window
    mhyp = [spkr[w.mean(axis=0).argmax()]
            for w in segment_axis(nhyp, win_size, win_size - 1, 0, 'cut')]

    res = pd.read_csv(os.path.join(
        out_dir, name + '_info.csv'), index_col=0)
    res['maw'] = mhyp
    res.drop('5-best', axis=1, inplace=True)
    res.to_csv(os.path.join(out_dir, name + '_info.csv'))
    return lium_score(name, out_dir, out_res=True,
                      hyp_col='maw', data_dir=args.data_path)


if __name__ == '__main__':
    main()
