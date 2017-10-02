#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# identity.py
# 2016 Marianna S. Buschle <msb@qtec.com>
#
# Simple identity element in python
#
# You can run the example from the source doing from gst-python/:
#
#  $ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/plugin:$PWD/examples/plugins
#  $ GST_DEBUG=python:4 gst-launch-1.0 fakesrc num-buffers=10 ! identity_py ! fakesink

from queue import Queue
from threading import Thread
import numpy as np
import soundfile as sf
from tempfile import mkstemp

import gi
gi.require_version('GstBase', '1.0')

from gi.repository import Gst, GObject, GstBase
Gst.init(None)


class SplitSilence(GstBase.BaseTransform):
    __gstmetadata__ = ('SplitSilence Python',
                       'Transform',
                       'Split buffer into files at silence',
                       'Cil Satriawan')

    _src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.caps_from_string('audio/x-raw'))

    _sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.caps_from_string('audio/x-raw'))

    __gsttemplates__ = (_src_template, _sink_template)

    def __init__(self):
        GstBase.BaseTransform.__init__(self)
        self.blk_q = Queue()
        self.sentinel = object()
        self.print_t = Thread(target=self.split_file, args=())
        self.print_t.start()

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
        sil_sum = 0
        split_thr = 4
        energy_thr = 0.01
        out_dir = '/home/cilsat/dev/scribe/test/splits'
        buf = []
        for raw in iter(self.blk_q.get, self.sentinel):
            blk = np.fromstring(raw, dtype=np.int16)
            rms = np.sqrt(np.mean((blk / 32768)**2))
            print(sil_sum, len(blk), rms)

            if rms < energy_thr:
                sil_sum += 1
                if not buf:
                    continue
                if sil_sum > split_thr:
                    fd, name = mkstemp(suffix='.wav', dir=out_dir)
                    sf.write(name, buf, samplerate=16000, subtype='PCM_16')
                    buf = []
            else:
                sil_sum = 0
            buf.extend(blk)


GObject.type_register(SplitSilence)
__gstelementfactory__ = ("splitsilence", Gst.Rank.NONE, SplitSilence)
