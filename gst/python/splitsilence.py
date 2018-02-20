#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# splitsilence.py
# 2017 Cil Satriawan <cil.satriawan@gmail.com>
#
# Split file/stream into separate files at silence
#
from queue import Queue
from threading import Thread
import numpy as np
import soundfile as sf
from detector import FrameDetector
from speechpy.feature import mfcc
from tempfile import mkstemp
import logging
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

logger = logging.getLogger(__name__)
Gst.init(None)
caps = 'audio/x-raw,format=S16LE,rate=16000,channels=1'


class GstPlugin(GstBase.BaseTransform):
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
        'split_thr': (
            GObject.TYPE_FLOAT,
            'Split Threshold',
            'Length of silence before splitting',
            0.1, 10.0, 0.3,  # min, max, default
            GObject.PARAM_READWRITE | GObject.PARAM_CONSTRUCT
        ),
        'energy_thr': (
            GObject.TYPE_FLOAT,
            'Silence Threshold',
            'Level (in dB) below which signal is considered silent',
            -60.0, 60.0, -15.0,  # min, max, default
            GObject.PARAM_READWRITE | GObject.PARAM_CONSTRUCT
        ),
        'out_dir': (
            GObject.TYPE_STRING,
            'Output directory',
            'Directory to store the resulting files',
            None,
            GObject.PARAM_READWRITE
        )
    }

    def __init__(self):
        GstBase.BaseTransform.__init__(self)

        self.samplerate = 16000
        self.props = {
            'split_thr': 0.3,
            'energy_thr': -15,
            'out_dir': '/tmp'
        }

        self.blk_q = Queue()
        self.sentinel = object()

        print_t = Thread(target=self.segment, args=())
        print_t.start()

    def set_property(self, prop, val):
        self.props[prop] = val

    def get_property(self, prop):
        return self.props[prop]

    def do_set_property(self, prop, val):
        if prop.name == 'split_thr':
            self.props['split_thr'] = val
        if prop.name == 'energy_thr':
            self.props['energy_thr'] = val
        if prop.name == 'blocksize':
            self.props['blocksize'] = val
        if prop.name == 'out_dir':
            self.props['out_dir'] = val

    def do_get_property(self, prop):
        val = None
        if prop.name == 'split_thr':
            val = self.props['split_thr']
        if prop.name == 'energy_thr':
            val = self.props['energy_thr']
        if prop.name == 'blocksize':
            val = self.props['blocksize']
        if prop.name == 'out_dir':
            val = self.props['out_dir']

    def do_transform_ip(self, buf):
        """
        This function is automatically called by Gst at each iteration. As the
        block data cannot be written back into stream, we must instead queue it
        and use it from a different thread.
        """
        # Gst.info("timestamp(buffer):%s" % (Gst.TIME_ARGS(buf.pts)))
        res, bmap = buf.map(Gst.MapFlags.READ)
        self.blk_q.put(bmap.data)

        return Gst.FlowReturn.OK

    def quit(self):
        self.blk_q.put(self.sentinel)

    def copy_data(self):
        with sf.SoundFile('copy.wav', mode='w', samplerate=16000,
                          channels=1, subtype='PCM_16') as f:
            for raw in iter(self.blk_q.get, self.sentinel):
                blk = np.fromstring(raw, dtype=np.int16)
                f.write(blk)

    def segment(self):
        """Gets audio blocks from stream and decides whether block is silent
        or non-silent. Accumulates non-silent blocks into a buffer until
        a specified number of silent blocks are detected, after which the
        buffer is dumped into a file.
        """
        energy_thr = 10**(0.1 *
                          self.props['energy_thr'])*np.iinfo(np.int16).max
        latency = self.samplerate
        sample_buffer = []
        out_segments = []
        # silence length threshold in samples
        sil_len_thr = int(self.props['split_thr'] * self.samplerate / 2048)

        fd = FrameDetector(self.samplerate, latency, 13, energy_thr,
                           sil_len_thr)

        for raw in iter(self.blk_q.get, self.sentinel):
            block = np.fromstring(raw, dtype=np.int16)
            sample_buffer.extend(block)
            fd.push_block(block)
            if fd.is_full():
                if is_silent(block) or is_turn():
                    out_segments.append(self.on_split(sample_buffer))
                    sample_buffer = []

        out_segments.append(self.on_split(sample_buffer))

    def on_split(self, sample_buffer):
        fd, name = mkstemp(suffix='.wav', dir=self.props['out_dir'])
        sf.write(name, sample_buffer, samplerate=self.samplerate,
                 subtype='PCM_16')
        logger.debug("%d %s" % (fd, name))
        return (fd, name)


GObject.type_register(GstPlugin)
__gstelementfactory__ = ("splitsilence", Gst.Rank.NONE, GstPlugin)
