#!/usr/bin/env python

import os
import sys
from multiprocessing import Pool, cpu_count
from subprocess import run, PIPE


def main():
    data_path = sys.argv[1]
    exp_path = sys.argv[2]
    diarize_sh = sys.argv[3]
    wavs = [n for n in os.listdir(data_path) if n.endswith('.lbl')]
    print(wavs)
    args = [(diarize_sh, os.path.join(data_path, w),
        os.path.join(exp_path, w.replace('.lbl', ''))) for w in wavs]
    with Pool(cpu_count()) as p:
        p.starmap(diarize, args)


def diarize(sh, wav, exp):
    if not os.path.exists(exp): os.mkdir(exp)
    with open(os.path.join(exp, 'log'), 'w') as f:
        run([sh, wav, exp], stderr=f)


if __name__ == "__main__":
    main()
