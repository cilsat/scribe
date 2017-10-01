#!/usr/bin/env python3
"""
Split input stream into files on silence.
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
from tempfile import mkstemp


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
    '-r', '--samplerate', type=int, default=44100, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, default='PCM_16',
    help='sound file subtype (e.g. "PCM_24")')
parser.add_argument(
    '-b', '--blocksize', type=int, default=1024,
    help='block size of stream')
parser.add_argument(
    '-s', '--split-thr', type=float, default=250,
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

    # stream()
    file_input()


def file_input(split_thr=args.split_thr, energy_thr=args.energy_thr,
               in_file=args.in_file, out_dir=args.out_dir,
               blocksize=args.blocksize):
    info = sf.info(in_file)
    base = os.path.basename(in_file).split('.')[0]
    samplerate = info.samplerate

    sil = True
    sil_sum = 0
    # number of silent blocks to tolerate before splitting
    split_thr = int(info.samplerate * split_thr / (1000 * blocksize))
    # energy cutoff
    energy_thr = 10**(0.1 * energy_thr)

    buf = []
    end = []
    dur = []
    for n, block in enumerate(sf.blocks(in_file, blocksize=blocksize)):
        rms = np.sqrt(np.mean(block**2))
        if rms < energy_thr:
            sil_sum += 1
            if not sil:
                sil = True

            if not buf:
                continue
            else:
                buf.extend(block)

            if sil_sum > split_thr:
                buf_l = int(100 * len(buf) / samplerate)
                print(buf_l)
                name = base + '_' + \
                    str(int(100 * blocksize * n / samplerate) - buf_l).zfill(6) \
                    + '_' + str(buf_l).zfill(3) + '.wav'
                sf.write(os.path.join(out_dir, name), buf,
                         samplerate=samplerate, subtype=info.subtype,
                         format=info.format)
                buf = []
        else:
            if sil:
                sil = False
                sil_sum = 0
            buf.extend(block)


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

                print(not sil, sil_sum, rms, len(buf))
                if rms < energy_thr:
                    sil_sum += 1
                    if not sil:
                        sil = True
                    if not buf:
                        continue
                    else:
                        buf.extend(block)

                    if sil_sum > split_thr:
                        print(np.array(buf))
                        fd, name = mkstemp(suffix='.wav', dir=args.out_dir)
                        sf.write(name, buf, samplerate=args.samplerate,
                                 subtype=args.subtype)
                        buf = []
                else:
                    if sil:
                        sil = False
                        sil_sum = 0
                    buf.extend(block)

    except KeyboardInterrupt:
        parser.exit(0)
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))


if __name__ == '__main__':
    main()
