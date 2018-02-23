import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.fftpack import dct
from .features import FrameGenerator

SIG_VOICED = 0
SIG_UNVOICED = 1
SIG_SPLIT = 2


class TurnDetector(object):
    def __init__(self, samplerate=16000, fb_size=20, num_cepstrals=13,
                 num_filters=40, energy_thr=-9.0, sil_len_thr=0.25,
                 block_size=2048):
        self.samplerate = samplerate
        # internal buffering
        self.fb = Buffer(fb_size, num_filters)
        self.sb = Buffer(10 * self.samplerate, num_cepstrals)

        # is_silent vars
        self.sil_sum = 0
        self.can_split = False
        self.energy_thr = energy_thr
        self.sil_len_thr = np.round(sil_len_thr * self.samplerate / block_size)

        # is_turn vars
        self.y = None
        self.cy = None
        self.ciy = None
        self.my = None

        self.fg = FrameGenerator(self.samplerate, num_filters=num_filters)

    def is_silent(self, block):
        rms = np.mean(np.abs(block))
        if rms < self.energy_thr:
            self.sil_sum += 1
            if self.can_split and self.sil_sum > self.sil_len_thr:
                self.can_split = False
                return True
        else:
            self.sil_sum = 0
            self.can_split = True
        return False

    def glr(self):
        if self.fb.is_full():
            return 0
        half = int(0.5 * self.fb.len)
        fx = self.fb.data[:half]
        mx = mn.logpdf(fx, np.mean(fx, axis=0), np.cov(fx, rowvar=False))
        fy = self.fb.data[half:]
        my = mn.logpdf(fy, np.mean(fy, axis=0), np.cov(fy, rowvar=False))
        mz = mn.logpdf(self.fb.data, np.mean(self.fb.data, axis=0),
                       np.cov(self.fb.data, rowvar=False))
        z = (mz.sum() - mx.sum() - my.sum()) / self.fb.len
        return z * 1.82

    def vad(self):
        if self.fb.is_full():
            return np.zeros((self.num_filters,))
        else:
            energy = np.mean(self.fb.data, axis=0)
            print(energy.mean())
            if energy.mean() < self.energy_thr:
                return energy
            else:
                return np.zeros((self.num_filters,))

    def is_voiced(self):
        if not self.fb.is_full():
            return SIG_UNVOICED
        elif np.mean(self.fb.data) < self.energy_thr:
            self.sil_sum += 1
            if self.can_split and self.sil_sum >= self.sil_len_thr:
                self.can_split = False
                return SIG_SPLIT
            else:
                return SIG_UNVOICED
        else:
            self.sil_sum = 0
            self.can_split = True
            return SIG_VOICED

    def is_turn(self):
        if self.sb.is_full():
            return SIG_SPLIT

        voiced = self.is_voiced()
        if voiced == SIG_VOICED:
            mfcc = dct(lmf, type=2, axis=-1,
                       norm='ortho')[:, :self.num_cepstrals]
            self.sb.push(mfcc)

    def push_block(self, block):
        lmf = self.fg.lmfe(block)
        self.fb.push(lmf)

        return self.is_voiced()
        voiced = self.is_voiced()
        if voiced == SIG_VOICED:
            mfcc = dct(lmf, type=2, axis=-1,
                       norm='ortho')[:, :self.num_cepstrals]
            self.sb.push(mfcc)
        elif voiced == SIG_SPLIT:
            self.sb.pop(self.fb.size)


class Buffer(object):
    """
    A class to manage the data in a fixed size buffer
    """

    def __init__(self, x, y=None):
        self.len = x
        if y:
            self.size = x * y
            self.data = np.empty((x, y))
        else:
            self.size = self.len
            self.data = np.empty((x,))
        self.idx = 0

    def push(self, samples):
        len_s = len(samples)
        if self.idx + len_s < self.len:
            self.data[self.idx:self.idx + len_s] = samples
            self.idx += len_s
        else:
            if self.idx == self.len:
                self.data[:-len_s] = self.data[len_s:]
            else:
                self.data[:-len_s] = self.data[len_s -
                                               self.len + self.idx:self.idx]
                self.idx = self.len
            self.data[-len_s:] = samples

    def pop(self, idx=None):
        if not idx:
            samples = np.copy(self.data[:self.idx])
            self.data[:] = np.empty(self.data.shape)
            self.idx = 0
        else:
            if idx > self.idx:
                raise ValueError()
            samples = np.copy(self.data[:idx])
            data = np.copy(self.data[idx:self.idx])
            self.data[:] = np.empty(self.data.shape)
            self.data[:self.idx - idx] = data
            self.idx -= idx
        return samples

    def is_full(self):
        return self.idx == self.len
