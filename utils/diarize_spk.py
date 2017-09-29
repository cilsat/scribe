#!/usr/bin/env python

import os
import sys
from multiprocessing import Pool, cpu_count
from subprocess import run, PIPE


def main():
    data_path = sys.argv[1]
    exp_path = sys.argv[2]
    diarize_sh = sys.argv[3]
    names = [n.split('.')[0] for n in os.listdir(data_path) if n.endswith('.lbl')]
    print(names)
    args = [(diarize_sh, os.path.join(data_path, n),
        os.path.join(exp_path, n)) for n in names]
    with Pool(cpu_count()) as p:
        p.starmap(diarize, args)


def diarize(sh, n, exp):
    if not os.path.exists(exp): os.mkdir(exp)
    with open(os.path.join(exp, 'log'), 'w') as f:
        if not os.path.exists(n+'.wav'):
            run(['ffmpeg', '-i', n+'.mp3', n+'t.wav'], stderr=f)
            run(['sox', n+'t.wav', n+'.wav', 'gain', '-6', 'highpass', '120'], stderr=f)
            os.remove(n+'t.wav')
        run([sh, n+'.wav', exp], stderr=f)


if __name__ == "__main__":
    main()
