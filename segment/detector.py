import numpy as np
from scipy.stats import multivariate_normal as mn
from ..feats.features import FrameGenerator


class FrameDetector(object):
    def __init__(self, samplerate, fb_size, num_coeffs, energy_thr,
                 sil_len_thr):
        self.samplerate = samplerate
        # internal buffering
        self.fb_size = fb_size*2
        self.fb = np.empty((self.fb_size, num_coeffs))
        self.fb.fill(np.nan)
        self.fb_idx = 0
        self.num_coeffs = num_coeffs

        self.sb = []

        # is_silent vars
        self.sil_sum = 0
        self.can_split = False
        self.energy_thr = energy_thr
        self.sil_len_thr = sil_len_thr

        # is_turn vars
        self.y = None
        self.cy = None
        self.ciy = None
        self.my = None

        self.fg = FrameGenerator(self.samplerate)

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
        if not self.is_full():
            return 0
        half = int(0.5 * self.fb_size)
        fx = self.fb[:half]
        mx = mn.logpdf(fx, np.mean(fx, axis=0), np.cov(fx, rowvar=False))
        fy = self.fb[half:]
        my = mn.logpdf(fy, np.mean(fy, axis=0), np.cov(fy, rowvar=False))
        mz = mn.logpdf(self.fb, np.mean(self.fb, axis=0), np.cov(self.fb,
                                                                 rowvar=False))
        z = (mz.sum() - mx.sum() - my.sum())/self.fb_size
        return z*1.82

    def vad(self):
        if not self.is_full():
            return np.zeros((self.num_coeffs,))
        else:
            energy = np.mean(self.fb, axis=0)
            if energy.mean() < self.energy_thr:
                return energy
            else:
                return np.zeros((self.num_coeffs,))

    def push_block(self, block):
        frames = self.fg.lmfe(block)

        if self.fb_idx + len(frames) < self.fb_size:
            self.fb[self.fb_idx:self.fb_idx + len(frames)] = frames
            self.fb_idx += len(frames)
        else:
            if self.fb_idx != self.fb_size:
                self.fb[:-len(frames)] = self.fb[len(frames) -
                                                 self.fb_size + self.fb_idx:self.fb_idx]
            else:
                self.fb[:-len(frames)] = self.fb[len(frames):]
            self.fb[-len(frames):] = frames
            self.fb_idx = self.fb_size
        return self.vad()

    def is_full(self):
        return self.fb_idx == self.fb_size
