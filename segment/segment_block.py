import numpy as np
import soundfile as sf
from speechpy.feature import mfcc
from tempfile import mkstemp


class Segmentor:
    def __init__(self, blocksize, samplerate, data_iter, out_dir):
        self.blocksize = blocksize
        self.samplerate = samplerate
        self.data_iter = data_iter
        self.out_dir = out_dir

        self.sample_buffer = []
        self.sil_sum = 0
        self.can_split = False

    def segment_block(self, is_silent=None, is_turn=None, on_split=None):
        if is_silent is None:
            is_silent = self.is_silent
        if is_turn is None:
            is_turn = self.is_turn
        if on_split is None:
            on_split = self.on_split

        energy_thr = 10**(0.1 * -20.0)
        print(energy_thr)
        buffer = []

        for n, block in enumerate(self.data_iter):
            # frame = mfcc(block, self.samplerate)

            if is_silent(block, energy_thr) or is_turn(block, buffer):
                print(len(buffer))
                on_split(buffer)
                buffer = []
            else:
                buffer.extend(block)
        on_split(buffer)

    def is_silent(self, block, energy_thr=10.0, sil_len_thr=3):
        rms = np.sqrt(np.mean(block**2))
        if rms < energy_thr:
            self.sil_sum += 1
            if self.can_split and self.sil_sum > sil_len_thr:
                print(rms)
                self.can_split = False
                return True
        else:
            self.sil_sum = 0
            self.can_split = True
        return False

    def is_turn(self, x, y, theta=1.82):
        frame_buffer = 

        return False

    def on_split(self, sample_buffer):
        fd, name = mkstemp(suffix='.wav', dir=self.out_dir)
        sf.write(name, sample_buffer, samplerate=self.samplerate,
                 subtype='PCM_16')
