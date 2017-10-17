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
import pandas as pd
from tempfile import mkstemp
from identify_spk import test
from lium_utils import lbl2df


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
args = parser.parse_args()


def main():
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    # if args.stream:
        # stream_input()
    # else:
        # file_input()

    wav_path = '/home/cilsat/data/speech/rapat'
    names = [n.split('.')[0] for n in os.listdir(wav_path)
             if n.endswith('.lbl')]

    paths = {n: os.path.join(args.out_dir, n) for n in names}

    for n in names:
        if not os.path.exists(paths[n]):
            os.mkdir(paths[n])

    for n in names:
        file_input(in_file=os.path.join(
            wav_path, n + '.wav'), out_dir=paths[n])

    for n in names:
        lium_test(name=n, out_dir=paths[n])

    for n in names:
        lium_score(name=n, out_dir=paths[n], data_dir=wav_path)


def file_input(split_thr=args.split_thr, energy_thr=args.energy_thr,
               in_file=args.in_file, out_dir=args.out_dir,
               blocksize=args.blocksize):
    info = sf.info(in_file)
    base = os.path.basename(in_file).split('.')[0]
    fill = int(np.log10(info.duration * 100))
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
                start = 100 * mult * n - dur
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
    lium = '/home/cilsat/down/prog/lium_spkdiarization-8.4.1.jar'
    gmm = '/home/cilsat/data/speech/rapat/90s_all/spk.gmm'
    ubm = '/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm'
    log = os.path.join(out_dir, name + '.log')
    for n in df.index:
        seg = os.path.join(out_dir, n.replace('.wav', '.uem.seg'))
        wav = os.path.join(out_dir, n)
        iseg = seg.replace('.uem.seg', '.i.seg')
        lbl = n.split('.')[0]
        test(lium=lium, seg=seg, wav=wav, iseg=iseg,
             gmm=gmm, ubm=ubm, name=lbl, log=log)

    # Get results of speaker identification and stuff them into info
    hyp = []
    for n in df.index:
        with open(os.path.join(out_dir, n.replace('.wav', '.i.seg'))) as f:
            hyp.append(int(f.read().split('#')[-1][1:-1]))

    df['hyp'] = hyp
    df.to_csv(info)


def lium_score(name, out_dir, data_dir='/home/cilsat/data/speech/rapat'):
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
    rights = []
    for key, hyp in df.iterrows():
        # get last ref segment that starts before hyp starts
        try:
            begin = ref.loc[ref.start <= hyp.start].iloc[-1]
        except Exception as e:
            print(hyp.name)
            begin = ref.iloc[0]
        # get first ref segment that ends after hyp ends
        try:
            end = ref.loc[ref.start + ref.dur >= hyp.start + hyp.dur].iloc[0]
        except Exception as e:
            print(hyp.name)
            end = ref.iloc[-1]
        # calculate how many frames hyp identifies correctly
        score = 0
        # if all of hyp is contained within one ref segment
        if begin.name == end.name:
            if begin.cls == hyp.hyp:
                score += hyp.dur
        else:
            if begin.cls == hyp.hyp:
                score += begin.start + begin.dur - hyp.start
            if end.cls == hyp.hyp:
                score += hyp.start + hyp.dur - end.start
            for n in range(begin.name + 1, end.name):
                if ref.loc[n, 'cls'] == hyp.hyp:
                    score += ref.loc[n, 'dur']
        scores.append(score)
        rights.append(begin.cls)

    df['score'] = scores
    df['right'] = rights
    df.score = df.score.astype(int)
    df.right = df.right.astype(int)
    df.to_csv(info)


if __name__ == '__main__':
    main()
