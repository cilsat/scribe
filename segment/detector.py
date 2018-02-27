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
from scipy.stats import multivariate_normal as mn
from scipy.fftpack import dct
from .features import FeatureGenerator
from .buffer import Buffer

SIG_VOICED = 0
SIG_UNVOICED = 1
SIG_SPLIT = 2

logger = logging.getLogger(__name__)


class TurnDetector(object):
    """Class that detects speaker turn changes."""

    def __init__(self, samplerate=16000, fb_size=20, num_cepstrals=13,
                 num_filters=40, energy_thr=6.0, sil_len_thr=0.25,
                 block_size=2048, blk_q=Queue(), out_dir='/tmp', on_split=None,
                 lium='/home/cilsat/net/Files/lium_spkdiarization-8.4.1.jar',
                 ubm='/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm',
                 gmm='/home/cilsat/data/speech/rapat/gmm/120_rapat.gmm'):
        """Initialize with samplerate and sizes of internal buffers."""
        # main loop
        self.samplerate = samplerate
        self.blk_q = blk_q
        self.sentinel = object()
        self.out_dir = out_dir
        self._on_split = on_split or self.on_split

        # feature attributes
        self.num_cepstrals = num_cepstrals
        self.num_filters = num_filters

        # internal buffering
        # VAD requires around 200 ms of buffered LMFE feature frames.
        self.vad_buf = Buffer(fb_size, num_filters)
        # Sample buffer of unlimited size to store samples in block queue.
        self.smp_buf = []
        # Turn buffer to analyze speaker changes
        self.turn_buf = Buffer(10 * self.samplerate, num_cepstrals)

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
            #logger.debug("Voiced: Buffer not full")
            return SIG_UNVOICED
        elif np.mean(self.vad_buf.data) < self.energy_thr:
            #logger.debug("Voiced: SIL")
            self.sil_sum += 1
            if self.can_split and self.sil_sum >= self.sil_len_thr:
                self.can_split = False
                return SIG_SPLIT
            else:
                return SIG_UNVOICED
        else:
            #logger.debug("Voiced: %s" % np.mean(self.vad_buf.data))
            self.sil_sum = 0
            self.can_split = True
            return SIG_VOICED

    def start(self):
        """Start processing loop in separate thread."""
        loop = asyncio.get_event_loop()
        logger.debug("Starting TurnDetector thread.")
        loop.run_until_complete(self.process_blocks(loop))
        loop.close()

    def stop(self):
        """Stop processing loop by putting sentinel."""
        logger.debug("Stopping TurnDetector thread.")
        self.blk_q.put(self.sentinel)

    async def process_blocks(self, loop):
        """Process contents of block queue until sentinel is found."""
        results = []
        for n, data in enumerate(iter(self.blk_q.get, self.sentinel)):
            timestamp, raw = data
            blk = np.fromstring(raw, dtype=np.int16)
            lmf = self.fg.lmfe(blk)
            self.vad_buf.push(lmf)
            #logger.debug("Pushed block %d of size %d into buffer at %s." % (n,
            #             len(lmf), timestamp))

            voiced = self.is_voiced()
            if voiced == SIG_VOICED:
                self.smp_buf.extend(blk)
                mfcc = dct(lmf, type=2, axis=-1,
                           norm='ortho')[:, :self.num_cepstrals]
                self.turn_buf.push(mfcc)
            elif voiced == SIG_SPLIT:
                result = self._on_split(n)
                results.append(result)

        results.append(self._on_split(n))
        return results

    def on_split(self, blk_id):
        """Write WAV with contents of sample buffer."""
        fd, name = mkstemp(prefix=str(blk_id).zfill(4), suffix='.wav',
                           dir=self.out_dir)
        sf.write(name, self.smp_buf, samplerate=self.samplerate,
                 subtype='PCM_16')

        num_frames = int(len(self.smp_buf) * 100 / self.samplerate)
        spk_hyp = self.identify_speaker(name, num_frames, self.lium, self.ubm,
                                        self.gmm)
        txt_hyp = self.recognize_speech(name,
                                        '/home/cilsat/src/kaldi/src',
                                        '/home/cilsat/src/kaldi-gstreamer-server/models')
        end_smp = blk_id * self.block_size
        start_smp = end_smp - len(self.smp_buf)
        result = json.dumps({
            'start': start_smp / self.samplerate,
            'end': end_smp / self.samplerate,
            'sid': spk_hyp,
            'text': txt_hyp,
        })
        logger.debug("%s" % result)

        self.smp_buf = []
        return result

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
            '--sOutputMask=' + fin_seg,
            '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
            '--tInputMask=' + gmm, '--sTop=5,' + ubm,
            '--sSetLabel=add', name
        ]
        with open(log_seg, 'a') as f:
            run(cmd, stderr=f)
        # Read results
        with open(fin_seg) as f:
            hyp = int(f.read().split('#')[-1][1:-1])
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

        #with open(scp, 'w') as f:
        #    f.write(name.split('.')[0] + ' ' + name)
        # Call gmm-latgen-faster
        cmd = [
            #compute_mfcc, 'scp:' + scp, 'ark:-|',
            #add_deltas, 'ark:-', 'ark:-|',
            #gmm_latgen_faster, '--word-symbol-table=' + txt, mdl, fst,
            #'ark:-', 'ark:/dev/null', 'ark,t:' + out
            './decode_utt.sh', name, model
        ]
        with open(log_asr, 'a') as f:
            run(cmd, stderr=f)
        with open(log_asr) as f:
            hyp = [n.replace(name, '') for n in f.read().splitlines() if
                   n.startswith(name)][0]
        return hyp

