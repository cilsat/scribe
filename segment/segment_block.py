import sys
import numpy as np
import soundfile as sf
from speechpy.feature import mfcc
from tempfile import mkstemp


class Detector(object):
    # statics for is_silent
    _sil_sum = 0
    _can_split = False
    # statics for is_turn
    _y = None
    _cy = None
    _ciy = None
    _my = None

    @classmethod
    def is_silent(cls, block, energy_thr=10.0, sil_len_thr=5):
        rms = np.sqrt(np.mean(block**2))
        if rms < energy_thr:
            cls._sil_sum += 1
            if cls._can_split and cls._sil_sum > sil_len_thr:
                cls._can_split = False
                return True
        else:
            cls._sil_sum = 0
            cls._can_split = True
        return False

    @classmethod
    def is_turn(cls, x, thr=1000, theta=1.82):
        if x is None or len(x) <= 1:
            print('x None')
            return False
        elif cls._y is None:
            print('y None')
            cls._y = x
            cls._cy = np.cov(cls._y, rowvar=False)
            cls._ciy = np.linalg.inv(cls._cy)
            cls._my = np.mean(cls._y, axis=0)
            return False
        else:
            cx = np.cov(x, rowvar=False)
            cix = np.linalg.inv(cx)
            mx = np.mean(x, axis=0)
            dxy = mx - cls._my
            kl2 = np.trace((cx - cls._cy) * (cls._ciy - cix)) + \
                np.trace((cls._ciy + cix) * np.outer(dxy, dxy))
            cls._y = x
            cls._cy = cx
            cls._ciy = cix
            cls._my = mx
            if kl2 > thr:
                print(kl2)
                return True


class Segmentor:
    def __init__(self, blocksize, samplerate, data_iter, out_dir):
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.data_iter = data_iter
        self.out_dir = out_dir

        self.sil_sum = 0
        self.can_split = False

    def segment_block(self, is_silent=Detector.is_silent,
                      is_turn=Detector.is_turn, on_split=None):
        if on_split is None:
            on_split = self.on_split

        energy_thr = 10**(0.1 * -20.0)
        sample_buffer = []

        for n, block in enumerate(self.data_iter):
            frames = mfcc(block, self.samplerate)
            sample_buffer.extend(block)

            if is_turn(frames):
                on_split(sample_buffer)
                sample_buffer = []

        on_split(sample_buffer)

    def on_split(self, sample_buffer):
        fd, name = mkstemp(suffix='.wav', dir=self.out_dir)
        sf.write(name, sample_buffer, samplerate=self.samplerate,
                 subtype='PCM_16')
