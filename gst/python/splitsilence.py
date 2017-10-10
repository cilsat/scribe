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
from tempfile import mkstemp

import gi
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

Gst.init(None)
caps = 'audio/x-raw,format=S16LE,rate=16000,channels=1'


class SplitSilence(GstBase.BaseTransform):
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
            GObject.TYPE_UINT,
            'Split Threshold',
            'Max consecutive silent blocks before gate',
            1, GObject.G_MAXUINT, 3,  # min, max, default
            GObject.PARAM_READWRITE | GObject.PARAM_CONSTRUCT
        ),
        'energy_thr': (
            GObject.TYPE_INT,
            'Silence Threshold',
            'Level (in dB) below which signal is considered silent',
            GObject.G_MININT, GObject.G_MAXINT, -15,  # min, max, default
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

        self.props = {
            'split_thr': 3,
            'energy_thr': -15,
            'out_dir': './splits'
        }
        self.samplerate = 16000
        self.blk_q = Queue()
        self.sentinel = object()
        self.print_t = Thread(target=self.split_file, args=())
        self.print_t.start()

    def do_set_property(self, prop, val):
        if prop.name == 'split_thr':
            self.props['split_thr'] = val
        if prop.name == 'energy_thr':
            self.props['energy_thr'] = val
        if prop.name == 'out_dir':
            self.props['out_dir'] = val

    def do_get_property(self, prop):
        val = None
        if prop.name == 'split_thr':
            val = self.props['split_thr']
        if prop.name == 'energy_thr':
            val = self.props['energy_thr']
        if prop.name == 'out_dir':
            val = self.props['out_dir']

    def do_transform_ip(self, buf):
        # Gst.info("timestamp(buffer):%s" % (Gst.TIME_ARGS(buf.pts)))
        res, map = buf.map(Gst.MapFlags.READ)
        assert res
        self.blk_q.put(map.data)

        return Gst.FlowReturn.OK

    def quit(self):
        self.blk_q.put(self.sentinel)

    def copy_data(self):
        with sf.SoundFile('copy.wav', mode='w', samplerate=16000,
                          channels=1, subtype='PCM_16') as f:
            for raw in iter(self.blk_q.get, self.sentinel):
                blk = np.fromstring(raw, dtype=np.int16)
                f.write(blk)

    def split_file(self):
        """
        Gets audio blocks from stream and decides whether block is silent
        or non-silent. Accumulates non-silent blocks into a buffer until
        a specified sequence of silent blocks is detected, after which
        buffer is dumped into a file.
        """
        # number of consecutive silent blocks detected so far
        sil_sum = 0
        # maximum sil_sum before buffer is dumped to file
        split_thr = self.props['split_thr']
        # level in dB below which a block is considered silent
        energy_thr = 10**(0.1 * self.props['energy_thr']) * 32768
        # directory to place split files
        out_dir = '/home/cilsat/dev/scribe/test/splits/m0002-0'
        # list to store audio buffer
        buf = []

        for raw in iter(self.blk_q.get, self.sentinel):
            # convert raw bytes to ndarray
            blk = np.fromstring(raw, dtype=np.int16)
            # estimate energy level of current block
            rms = np.mean(np.abs(blk))

            if rms < energy_thr:
                sil_sum += 1
                if not buf:
                    continue
                else:
                    buf.extend(blk)
                if sil_sum > split_thr:
                    fd, name = mkstemp(suffix='.wav', dir=out_dir)
                    sf.write(name, buf, samplerate=16000, subtype='PCM_16')
                    print(len(buf), fd)
                    buf = []
            else:
                sil_sum = 0
                buf.extend(blk)


GObject.type_register(SplitSilence)
__gstelementfactory__ = ("splitsilence", Gst.Rank.NONE, SplitSilence)
