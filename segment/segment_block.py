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
        rms = np.mean(np.abs(block))
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
    def is_turn(cls, y, func, thr):
        if y is None:
            return False
        elif cls.x is None:
            cls.x = y
        else:
            return func(x, y) < thr


class Segmentor:
    def __init__(self, data_iter, out_dir, blocksize=800, samplerate=16000):
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.data_iter = data_iter
        self.out_dir = out_dir

        self.sil_sum = 0
        self.can_split = False

    def segment_block(self, is_silent=None,
                      is_turn=None, on_split=None,
                      energy_thr=-10, sil_len_thr=0.75):
        if on_split is None:
            on_split = self.on_split
        if is_silent is None:
            is_silent = Detector.is_silent
        if is_turn is None:
            is_turn = Detector.is_turn

        energy_thr = 10**(0.1 * energy_thr)
        sil_len_thr = int(sil_len_thr * self.samplerate / self.blocksize)
        sample_buffer = []
        out_segments = []

        for n, block in enumerate(self.data_iter):
            frames = mfcc(block, self.samplerate)
            sample_buffer.extend(block)

            if is_silent(block, energy_thr, sil_len_thr):
                out_segments.append(on_split(sample_buffer))
                sample_buffer = []

        out_segments.append(on_split(sample_buffer))
        return out_segments

    def on_split(self, sample_buffer):
        fd, name = mkstemp(suffix='.wav', dir=self.out_dir)
        sf.write(name, sample_buffer, samplerate=self.samplerate,
                 subtype='PCM_16')
        return (fd, name)
