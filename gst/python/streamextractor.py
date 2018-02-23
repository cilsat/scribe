#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# splitsilence.py
# 2017 Cil Satriawan <cil.satriawan@gmail.com>
#
# Split file/stream into separate files at silence
#
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase
from queue import Queue

Gst.init(None)


class GstPlugin(GstBase.BaseTransform):
    __gstmetadata__ = ('Stream Extractor',
                       'Transform',
                       'Get GST buffer and push into python queue',
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

    def __init__(self, blk_q=Queue()):
        GstBase.BaseTransform.__init__(self)
        self.blk_q = blk_q

    def do_transform_ip(self, buf):
        """
        This function is automatically called by Gst at each iteration. As the
        block data cannot be written back into stream, we must instead queue it
        and use it from a different thread.
        """
        timestamp = Gst.TIME_ARGS(buf.pts)
        Gst.info("timestamp(buffer):%s  queue(buffer):%s" % (timestamp,
                                                             self.blk_q.qsize()))
        res, bmap = buf.map(Gst.MapFlags.READ)
        self.blk_q.put((timestamp, bmap.data))

        return Gst.FlowReturn.OK


GObject.type_register(GstPlugin)
__gstelementfactory__ = ("streamextractor", Gst.Rank.NONE, GstPlugin)
