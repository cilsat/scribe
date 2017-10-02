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


class RemoveSilence(GstBase.BaseTransform):
    __gstmetadata__ = ('RemoveSilence Python',
                       'Transform',
                       'Simple identity element written in python',
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
        self.print_t = Thread(target=self.copy_data, args=())
        self.print_t.start()

    def do_transform_ip(self, buf):
        # Gst.info("timestamp(buffer):%s" % (Gst.TIME_ARGS(buf.pts)))
        res, map = buf.map(Gst.MapFlags.READ)
        assert res
        self.blk_q.put(map.data)

        return Gst.FlowReturn.OK

    def copy_data(self):
        with sf.SoundFile('copy.wav', mode='w', samplerate=16000,
                          channels=1, subtype='PCM_16') as f:
            for raw in iter(self.blk_q.get, None):
                blk = np.fromstring(raw, dtype=np.int16)
                f.write(blk)

    def split_file(self):
        sil = True
        sil_sum = 0

        split_thr = 2
        energy_thr = 10.0

        buf = []
        while True:
            blk = self.blk_q.get().astype(np.float)
            rms = np.sqrt(np.mean((blk / 32768)**2))
            print(rms)

            if rms < energy_thr:
                sil_sum += 1
                if not sil:
                    sil = True
                if not buf:
                    continue
                else:
                    buf.extend(blk)

                if sil_sum > split_thr:
                    fd, name = mkstemp(
                        suffix='.wav', dir='/home/cilsat/dev/test/splits')
                    sf.write(name, buf, samplerate=16000, subtype='PCM_16')
                    buf = []
            else:
                if sil:
                    sil = False
                    sil_sum = 0
                buf.extend(blk)


GObject.type_register(RemoveSilence)
__gstelementfactory__ = ("removesilence", Gst.Rank.NONE, RemoveSilence)
