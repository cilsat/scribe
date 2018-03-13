"""Module that implements the TurnDetector class."""
import os
import numpy as np
import soundfile as sf
import logging
import asyncio
import json

from subprocess import run
from threading import Thread
from tempfile import mkstemp
from queue import Queue
from time import perf_counter
from scipy.stats import multivariate_normal as mn
from scipy.fftpack import dct
from scipy.linalg import det, pinv
from .features import FeatureGenerator
from .buffer import Buffer

SIG_VOICED = 0
SIG_UNVOICED = 1
SIG_SIL = 2
SIG_SAME = 3
SIG_TURN = 4

logger = logging.getLogger(__name__)


class TurnDetector(object):
    """Class that detects speaker turn changes."""

    def __init__(self, samplerate=16000, fb_size=20, num_cepstrals=13,
                 num_filters=40, energy_thr=6.0, sil_len_thr=0.25,
                 block_size=2048, out_dir='/tmp', on_sil=None, on_split=None,
                 lium='/home/cilsat/net/Files/lium_spkdiarization-8.4.1.jar',
                 ubm='/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm',
                 gmm='/home/cilsat/data/speech/rapat/gmm/120_rapat.gmm',
                 loop=None, on_result=None):
        """Initialize with samplerate and sizes of internal buffers."""
        # main loop
        self.samplerate = samplerate
        self.blk_q = asyncio.Queue(loop=loop)
        self.sentinel = object()
        self.out_dir = out_dir
        self._on_split = on_split or self.on_split
        self._on_result = on_result or self.on_result

        # feature attributes
        self.num_cepstrals = num_cepstrals
        self.num_filters = num_filters

        # internal buffering
        # VAD requires around 200 ms of buffered LMFE feature frames.
        self.vad_buf = Buffer(int(0.005 * self.samplerate), num_filters)
        # Sample buffer of unlimited size to store samples in block queue.
        self.smp_buf = []
        # Frame buffer of unlimited size to store frames of corresponding
        # sample buffer.
        self.turn_buf = Buffer(
            int(0.02 * self.samplerate), num_cepstrals)
        self.cur_turn = Buffer(
            int(0.02 * self.samplerate), num_cepstrals)
        self.prev_turn = Buffer(
            int(0.02 * self.samplerate), num_cepstrals)

        # is_silent vars
        self.sil_sum = 0
        self.can_split = False
        self.energy_thr = energy_thr
        self.sil_len_thr = np.round(sil_len_thr * self.samplerate / block_size)
        self.block_size = block_size

        # is_turn vars
        self.y = None
        self.cy = None
        self.ciy = None
        self.my = None

        # speaker identification
        self.lium = lium
        self.ubm = ubm
        self.gmm = gmm

        self.fg = FeatureGenerator(self.samplerate, num_filters=num_filters)

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

    def glr(self, frame):
        fx = self.turn_buf.get_data()
        mx = mn.logpdf(fx, np.mean(fx, axis=0), np.cov(fx, rowvar=False))
        fy = frame
        my = mn.logpdf(fy, np.mean(fy, axis=0), np.cov(fy, rowvar=False))
        fz = np.vstack((self.turn_buf.data, frame))
        mz = mn.logpdf(fz, np.mean(fz, axis=0), np.cov(fz, rowvar=False))
        z = (mz.mean() - mx.mean() - my.mean()) / len(fz)
        logger.debug("Lengths: %d %d %d" % (len(fx), len(fy), len(fz)))
        return z

    def glr2(self, data):
        half = int(0.5 * len(data))
        fx = data[:half]
        fy = data[half:]
        cx = np.cov(fx, rowvar=False)
        cy = np.cov(fy, rowvar=False)
        nx = len(fx)
        ny = len(fy)
        n = nx + ny
        d1 = -0.5 * (nx * np.log(det(cx)) + ny * np.log(det(cy)))
        d2 = np.log(det((nx * cx + ny * cy) / n))
        return d1 - d2

    def kl2(self, data):
        half = int(0.5 * len(data))
        fx = data[:half]
        fy = data[half:]
        cx = np.cov(fx, rowvar=False)
        cy = np.cov(fy, rowvar=False)
        dmxy = np.mean(fx, axis=0) - np.mean(fy, axis=0)
        ix = pinv(cx)
        iy = pinv(cy)
        d = 0.5 * (np.trace((cx - cy) * (iy - ix)) +
                   np.trace((ix + iy) * dmxy * dmxy.T))
        return d

    def bic(self, fx, fy, lambdac=1.4):
        fz = np.vstack((fx, fy))
        cx = np.cov(fx, rowvar=False)
        cy = np.cov(fy, rowvar=False)
        cz = np.cov(fz, rowvar=False)
        mx = len(fx) * np.log(det(cx))
        my = len(fy) * np.log(det(cy))
        mz = len(fz) * np.log(det(cz))
        d = 0.5 * (mz - mx - my)
        p = fz.shape[1]
        d -= lambdac * 0.5 * (p + 0.5 * p * (p + 1)) * np.log(len(fz))
        return d

    def is_voiced(self, blk):
        """Detect whether current contents of frame buffer is voiced."""
        self.vad_buf.push(self.fg.lmfe(blk))
        # logger.debug("%f", np.mean(self.vad_buf.data))
        if not self.vad_buf.is_full():
            # logger.debug("Voiced: Buffer not full")
            return SIG_UNVOICED
        elif np.mean(self.vad_buf.data) < self.energy_thr:
            # logger.debug("Voiced: SIL")
            self.sil_sum += 1
            if self.can_split and self.sil_sum >= self.sil_len_thr:
                self.can_split = False
                return SIG_SIL
            else:
                return SIG_UNVOICED
        else:
            self.sil_sum = 0
            self.can_split = True
            return SIG_VOICED

    def is_turn(self, blk):
        """Detect whether turn buffer contains speech from more than 1 speaker.
        """
        self.turn_buf.push(self.fg.mfcc(blk, use_energy=False))
        if not self.turn_buf.is_full():
            return SIG_SAME
        kl2 = self.kl2(self.turn_buf.data)
        glr2 = self.glr2(self.turn_buf.data)
        logger.debug("KL2: %f GLR2: %f" % (kl2, glr2))
        if kl2 > 10:
            self.turn_sum += 1
        # if self.cur_turn.idx >= self.num_cepstrals:
        #    if not self.prev_turn.is_empty():
        #        bic = self.bic(self.prev_turn.get_data(),
        #                       self.cur_turn.get_data())
        #        logger.debug("BIC: %f" % bic)
        #        if bic > 0:
        #            # self._on_split(n, self.smp_buf)
        #            self.smp_buf = []
        #            self.prev_turn.pop()
        #    self.prev_turn.push(self.cur_turn.pop())

    def start(self):
        """Start processing loop in separate thread."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.process_blocks(loop))
        loop.close()

    async def stop(self):
        """Stop processing loop by putting sentinel."""
        logger.debug("Stopping TurnDetector thread.")
        await self.blk_q.put(self.sentinel)

    async def process_blocks(self):
        """Process contents of block queue until sentinel is found."""
        logger.debug("Starting TurnDetector loop.")
        results = []
        n = 0
        while True:
            data = await self.blk_q.get()
            if data == self.sentinel:
                break
            await self.process_block(data)
            n += 1
            self.blk_q.task_done()

    def process_block(self, block, last=False):
        if last and len(self.smp_buf) > 0:
            self._on_split(np.array(self.smp_buf))
        data = np.fromstring(block, dtype=np.int16)
        voiced = self.is_voiced(data)
        if voiced == SIG_VOICED:
            self.smp_buf.extend(data)
            self.cur_turn.push(self.fg.mfcc(data, use_energy=False))
        elif voiced == SIG_SIL:
            self._on_split(np.array(self.smp_buf))
            self.smp_buf = []

    def on_split(self, samples):
        """Define what to do when a low energy segment is detected."""
        # Write samples to WAV.
        fd, name = mkstemp(suffix='.wav', dir=self.out_dir)
        sf.write(name, samples, samplerate=self.samplerate,
                 subtype='PCM_16')

        num_frames = int(len(samples) * 100 / self.samplerate)
        spk_hyp = self.identify_speaker(name, num_frames, self.lium, self.ubm,
                                        self.gmm)
        self._on_result(spk_hyp)

    def on_result(self, result):
        logger.debug("Received result: %s" % result)

    @staticmethod
    def identify_speaker(name, num_frames, lium, ubm, gmm):
        """Identify speakers with LIUM (quick and dirty)."""
        init_seg = name.replace('.wav', '.uem.seg')
        fin_seg = name.replace('.wav', '.seg')
        log_seg = name.replace('.wav', '.seg.log')
        # Prepare initial segmentation file
        with open(init_seg, 'w') as f:
            f.write(name + ' 1 0 ' + str(num_frames) + ' U U U S1')
        # Call LIUM
        cmd = [
            'java', '-cp', lium,
            'fr.lium.spkDiarization.programs.Identification',
            '--sInputMask=' + init_seg, '--fInputMask=' + name,
            '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
            '--sOutputMask=' + fin_seg,
            '--tInputMask=' + gmm, '--sTop=5,' + ubm,
            '--sSetLabel=add', name
        ]
        with open(log_seg, 'a') as f:
            run(cmd, stderr=f)
        # Read results
        with open(fin_seg) as f:
            hyp = f.read().split('#')[-1][:-1]
        return hyp

    @staticmethod
    def recognize_speech(name, kaldi, model):
        """Recognize speech with Kaldi through Gst."""
        scp = name.replace('.wav', '.scp')
        out = name.replace('.wav', '.txt')
        log_asr = name.replace('.wav', '.asr.log')

        compute_mfcc = os.path.join(kaldi, 'featbin/compute-mfcc-feats')
        add_deltas = os.path.join(kaldi, 'featbin/add-deltas')
        gmm_latgen_faster = os.path.join(kaldi, 'gmmbin/gmm-latgen-faster')

        # with open(scp, 'w') as f:
        #    f.write(name.split('.')[0] + ' ' + name)
        # Call gmm-latgen-faster
        cmd = [
            # compute_mfcc, 'scp:' + scp, 'ark:-|',
            # add_deltas, 'ark:-', 'ark:-|',
            # gmm_latgen_faster, '--word-symbol-table=' + txt, mdl, fst,
            # 'ark:-', 'ark:/dev/null', 'ark,t:' + out
            './decode_utt.sh', name, model
        ]
        with open(log_asr, 'a') as f:
            run(cmd, stderr=f)
        with open(log_asr) as f:
            hyp = [n.replace(name, '') for n in f.read().splitlines() if
                   n.startswith(name)][0]
        return hyp
