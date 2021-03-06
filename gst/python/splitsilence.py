"""Module that implements custom GstPlugin for speaker change detection."""
#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# splitsilence.py
# 2017 Cil Satriawan <cil.satriawan@gmail.com>
#
# Split file/stream into separate files at silence
#
import numpy as np
import soundfile as sf
from queue import Queue
from threading import Thread
from tempfile import mkstemp
from subprocess import run
from scribe.segment.detector import TurnDetector
import logging
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

logger = logging.getLogger(__name__)
Gst.init(None)
caps = 'audio/x-raw,format=S16LE,rate=16000,channels=1'


class GstPlugin(GstBase.BaseTransform):
    """Class that overrides Gst Plugin instantiation."""

    __gstmetadata__ = ('SplitSilence Python',
                       'Transform',
                       'Split buffer into files at silence',
                       'Cil Satriawan')

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.caps_from_string(caps)),
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.caps_from_string(caps))
    )

    __gproperties__ = {
        'split-thr': (
            GObject.TYPE_FLOAT,
            'Split Threshold',
            'Length of silence before splitting',
            0.1, 10.0, 0.3,  # min, max, default
            GObject.PARAM_READWRITE | GObject.PARAM_CONSTRUCT
        ),
        'energy-thr': (
            GObject.TYPE_FLOAT,
            'Silence Threshold',
            'Level (in dB) below which signal is considered silent',
            -60.0, 60.0, -15.0,  # min, max, default
            GObject.PARAM_READWRITE | GObject.PARAM_CONSTRUCT
        ),
        'out-dir': (
            GObject.TYPE_STRING,
            'Output directory',
            'Directory to store the resulting files',
            None,
            GObject.PARAM_READWRITE
        ),
        'lium': (
            GObject.TYPE_STRING,
            'Path to LIUM JAR',
            'Path to LIUM JAR',
            '/home/cilsat/net/Files/lium_spkdiarization-8.4.1.jar',
            GObject.PARAM_READWRITE
        ),
        'ubm': (
            GObject.TYPE_STRING,
            'Path to UBM',
            'Path to a UBM in alize format',
            '/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm',
            GObject.PARAM_READWRITE
        ),
        'gmm': (
            GObject.TYPE_STRING,
            'Path to GMM',
            'Path to speaker model in LIUM GMM format',
            '/home/cilsat/data/speech/rapat/spk.gmm',
            GObject.PARAM_READWRITE
        )
    }

    def __init__(self):
        """Initialize GstPlugin class."""
        GstBase.BaseTransform.__init__(self)

        self.samplerate = 16000
        self.split_thr = 0.3
        self.energy_thr = 15.0
        self.out_dir = "/tmp"

        self.lium = '/home/cilsat/down/prog/lium_spkdiarization-8.4.1.jar'
        self.ubm = '/home/cilsat/src/kaldi-offline-transcriber/models/ubm.gmm'
        self.gmm = '/home/cilsat/data/speech/rapat/gmm/120s_all_r2/spk.gmm'

        self.blk_q = Queue()
        self.sentinel = object()

        print_t = Thread(target=self.segment, args=())
        print_t.start()

    def do_stop(self):
        """Call on stop."""
        self.blk_q.put(self.sentinel)
        return

    def do_set_property(self, prop, val):
        """Set plugin properties."""
        if prop.name == 'split-thr':
            self.split_thr = val
        elif prop.name == 'energy-thr':
            self.energy_thr = val
        elif prop.name == 'out-dir':
            self.out_dir = val
        elif prop.name == 'lium':
            self.lium = val
        elif prop.name == 'ubm':
            self.ubm = val
        elif prop.name == 'gmm':
            self.gmm = val
        else:
            raise AttributeError("Unknown property %s" % prop.name)

    def do_get_property(self, prop):
        """Get plugin properties."""
        val = None
        if prop.name == 'split-thr':
            val = self.split_thr
        elif prop.name == 'energy-thr':
            val = self.energy_thr
        elif prop.name == 'out-dir':
            val = self.out_dir
        elif prop.name == 'lium':
            val = self.lium
        elif prop.name == 'ubm':
            val = self.ubm
        elif prop.name == 'gmm':
            val = self.gmm
        else:
            raise AttributeError("Unknown property %s" % prop.name)
        return val

    def do_transform_ip(self, buf):
        """
        Call each time buffer is received.

        As block data cannot be written back into the stream, we must instead
        queue it and use it from a different thread.
        """
        logger.debug("timestamp(buffer):%s" % (Gst.TIME_ARGS(buf.pts)))
        res, bmap = buf.map(Gst.MapFlags.READ)
        self.blk_q.put(bmap.data)

        return Gst.FlowReturn.OK

    def copy_data(self):
        """Write buffered data into temporary WAV file."""
        with sf.SoundFile('copy.wav', mode='w', samplerate=16000,
                          channels=1, subtype='PCM_16') as f:
            for raw in iter(self.blk_q.get, self.sentinel):
                blk = np.fromstring(raw, dtype=np.int16)
                f.write(blk)

    def segment(self):
        """
        Get audio blocks from stream and decides whether block is silent.

        Accumulates non-silent blocks into a buffer until
        a specified number of silent blocks are detected, after which the
        buffer is dumped into a file.
        """
        sample_buffer = []
        out_segments = []
        # TODO set properties through Gst getter setter
        # silence length threshold in samples
        sil_len_thr = self.split_thr
        energy_thr = self.energy_thr
        # Number of frames to buffer within detector
        buffer_size = 20
        # Latency due to buffering, in samples
        latency = int(buffer_size * 0.01 * self.samplerate)

        td = TurnDetector(self.samplerate, fb_size=buffer_size,
                          num_cepstrals=13, num_filters=20,
                          energy_thr=energy_thr, sil_len_thr=sil_len_thr)

        for n, raw in enumerate(iter(self.blk_q.get, self.sentinel)):
            blk = np.fromstring(raw, dtype=np.int16)
            voiced = td.push_block(blk)
            if voiced == 2:
                out = self.on_split(n, sample_buffer[:-latency])
                out_segments.append(out)
                sample_buffer = sample_buffer[-latency:]
            elif voiced == 0:
                sample_buffer.extend(blk)

    def on_split(self, blk_id, sample_buffer):
        """Call on split."""
        fd, name = mkstemp(prefix=str(blk_id).zfill(
            4), suffix='.wav', dir=self.out_dir)
        sf.write(name, sample_buffer, samplerate=self.samplerate,
                 subtype='PCM_16')

        # Speaker Identification
        frame_num = int(len(sample_buffer) * 100 / self.samplerate)
        hyp = self.identify_speaker(name, frame_num)
        logger.debug("%d %s %s" % (fd, name, hyp))

        return (fd, name)

    def identify_speaker(self, name, sample_length):
        """Quick and dirty speaker identification with LIUM."""
        init_seg = name.replace('.wav', '.uem.seg')
        fin_seg = name.replace('.wav', '.seg')
        log_seg = name.replace('.wav', '.log')
        # Prepare initial segmentation file
        with open(init_seg, 'w') as f:
            f.write(name + ' 1 0 ' + str(sample_length) + ' U U U S1')
        # Call LIUM
        cmd = [
            'java', '-cp', self.lium,
            'fr.lium.spkDiarization.programs.Identification',
            '--sInputMask=' + init_seg, '--fInputMask=' + name,
            '--sOutputMask=' + fin_seg,
            '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
            '--tInputMask=' + self.gmm, '--sTop=5,' + self.ubm,
            '--sSetLabel=add', name
        ]
        with open(log_seg, 'a') as f:
            run(cmd, stderr=f)
        # Read results
        with open(fin_seg) as f:
            hyp = int(f.read().split('#')[-1][1:-1])
        return hyp


GObject.type_register(GstPlugin)
__gstelementfactory__ = ("splitsilence", Gst.Rank.NONE, GstPlugin)

