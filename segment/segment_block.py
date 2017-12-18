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
    def is_silent(cls, block, energy_thr, sil_len_thr):
        # rms = np.sqrt(np.mean(block**2))
        rms = np.mean(block)
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
    def is_turn_generic(cls, y, func, thr):
        if y is None:
            return False
        elif cls.x is None:
            cls.x = y
        else:
            return func(x, y) < thr


class Segmentor:
    def __init__(self, blocksize, samplerate, data_iter, out_dir):
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.data_iter = data_iter
        self.out_dir = out_dir

        self.sil_sum = 0
        self.can_split = False

    def segment_block(self, is_silent=Detector.is_silent,
                      is_turn=Detector.is_turn, on_split=None,
                      energy_thr=-10, sil_len_thr=0.75):
        if on_split is None:
            on_split = self.on_split

        # energy_thr = 10**(0.1 * energy_thr)
        sil_len_thr = int(sil_len_thr * self.samplerate / self.blocksize)
        print(sil_len_thr)
        sample_buffer = []

        for n, block in enumerate(self.data_iter):
            frames = mfcc(block, self.samplerate)
            sample_buffer.extend(block)

            # if is_silent(frames[:, 0], energy_thr, sil_len_thr):
            if Detector.is_turn_generic(frames, is_turn, thr):
                print(n)
                on_split(sample_buffer)
                sample_buffer = []

        on_split(sample_buffer)

    def on_split(self, sample_buffer):
        fd, name = mkstemp(suffix='.wav', dir=self.out_dir)
        sf.write(name, sample_buffer, samplerate=self.samplerate,
                 subtype='PCM_16')
