import numpy as np
from speechpy.feature import mfcc
from scipy.stats import multivariate_normal as mn


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

    def push_block(self, block):
        frames = mfcc(block, sampling_frequency=self.samplerate,
                      frame_stride=0.01, num_cepstral=self.num_coeffs,
                      dc_elimination=False)

        len_f = len(frames)

        if self.fb_idx + len_f < self.fb_size:
            self.fb[self.fb_idx:self.fb_idx + len_f] = frames
            self.fb_idx += len_f
        else:
            if self.fb_idx != self.fb_size:
                src = len_f - self.fb_size + self.fb_idx
                self.fb[:-len_f] = self.fb[src:self.fb_idx]
            else:
                self.fb[:-len_f] = self.fb[len_f:]
            self.fb[-len_f:] = frames
            self.fb_idx = self.fb_size

    def is_full(self):
        return self.fb_idx == self.fb_size
