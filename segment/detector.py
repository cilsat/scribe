"""Module that implements the TurnDetector class."""
import numpy as np
import soundfile as sf
import logging

from threading import Thread
from tempfile import mkstemp
from queue import Queue
from scipy.stats import multivariate_normal as mn
from scipy.fftpack import dct
from .features import FrameGenerator
from .buffer import Buffer

SIG_VOICED = 0
SIG_UNVOICED = 1
SIG_SPLIT = 2

logger = logging.getLogger(__name__)


class TurnDetector(object):
    """Class that detects speaker turn changes."""

    def __init__(self, samplerate=16000, fb_size=20, num_cepstrals=13,
                 num_filters=40, energy_thr=-9.0, sil_len_thr=0.25,
                 block_size=2048, blk_q=Queue(), out_dir='/tmp'):
        """Initialize with samplerate and sizes of internal buffers."""
        # main loop
        self.samplerate = samplerate
        self.blk_q = blk_q
        self.sentinel = object()
        self.out_dir = out_dir

        # internal buffering
        # VAD requires around 200 ms of buffered LMFE feature frames.
        self.vad_buf = Buffer(fb_size, num_filters)
        # Sample buffer of unlimited size to store samples in block queue.
        self.smp_buf = []
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
        """Detect whether given block is silent."""
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
        """Calculate GLR of current contents of frame buffer."""
        if self.vad_buf.is_full():
            return 0
        half = int(0.5 * self.vad_buf.len)
        fx = self.vad_buf.data[:half]
        mx = mn.logpdf(fx, np.mean(fx, axis=0), np.cov(fx, rowvar=False))
        fy = self.vad_buf.data[half:]
        my = mn.logpdf(fy, np.mean(fy, axis=0), np.cov(fy, rowvar=False))
        mz = mn.logpdf(self.vad_buf.data, np.mean(self.vad_buf.data, axis=0),
                       np.cov(self.vad_buf.data, rowvar=False))
        z = (mz.sum() - mx.sum() - my.sum()) / self.vad_buf.len
        return z * 1.82

    def is_voiced(self):
        """Detect whether current contents of frame buffer is voiced."""
        if not self.vad_buf.is_full():
            return SIG_UNVOICED
        elif np.mean(self.vad_buf.data) < self.energy_thr:
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

    def push_block(self, block):
        """Push block from stream into frame and sample buffers."""
        lmf = self.fg.lmfe(block)
        self.vad_buf.push(lmf)

        return self.is_voiced()
        voiced = self.is_voiced()
        if voiced == SIG_VOICED:
            mfcc = dct(lmf, type=2, axis=-1,
                       norm='ortho')[:, :self.num_cepstrals]
            self.sb.push(mfcc)
        elif voiced == SIG_SPLIT:
            self.sb.pop(self.vad_buf.size)

    def start(self):
        """Start processing loop in separate thread."""
        self.loop_thread = Thread(target=self.loop)
        logger.debug("Starting TurnDetector thread.")
        self.loop_thread.start()

    def stop(self):
        """Stop processing loop by putting sentinel."""
        logger.debug("Stopping TurnDetector thread.")
        self.blk_q.put(self.sentinel)

    def loop(self):
        """Process contents of block queue until sentinel is found."""
        out_segments = []
        for n, data in enumerate(iter(self.blk_q.get, self.sentinel)):
            timestamp, raw = data
            blk = np.fromstring(raw, dtype=np.int16)
            lmf = self.fg.lmfe(blk)
            self.vad_buf.push(lmf)
            logger.debug("Pushed block %d of size %d into buffer at %s." % (n,
                         len(lmf), timestamp))

            voiced = self.is_voiced()
            if voiced == SIG_VOICED:
                self.smp_buf.extend(blk)
                mfcc = dct(lmf, type=2, axis=-1,
                           norm='ortho')[:, :self.num_cepstrals]
                self.sb.push(mfcc)
            elif voiced == SIG_SPLIT:
                fd, name = self.on_split(n)
                print(fd, name)
                out_segments.append((fd, name))

    def on_split(self, blk_id):
        """Write WAV with contents of sample buffer."""
        fd, name = mkstemp(prefix=str(blk_id).zfill(4), suffix='.wav',
                           dir=self.out_dir)
        sf.write(name, self.smp_buf, samplerate=self.samplerate,
                 subtype='PCM_16')
        return (fd, name)

