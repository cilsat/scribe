"""Module that implements the Stream Extractor pluign."""
#!/usr/bin/env python
# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

# splitsilence.py
# 2017 Cil Satriawan <cil.satriawan@gmail.com>
#
# Split file/stream into separate files at silence
#
import logging
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase
Gst.init(None)

logger = logging.getLogger(__name__)
caps = "audio/x-raw,format=S16LE,rate=(int)16000,channels=(int)1"


class StreamExtractor(GstBase.BaseTransform):
    """GstPlugin that extracts blocks from Gst pipeline into a queue."""

    __gstmetadata__ = ('Stream Extractor',
                       'Transform',
                       'Get GST buffer and push into python queue',
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
            Gst.caps_from_string(caps)))

    def __init__(self, blk_q=None):
        """Initialize the Gst Plugin."""
        GstBase.BaseTransform.__init__(self)
        self.blk_q = blk_q
        logger.debug("Initialize GstPlugin.")

    def do_transform_ip(self, buf):
        """
        Call each time buffer is received from pipeline.

        As block data cannot be written back into stream, we must instead
        queue it and use it from a different thread.
        """
        timestamp = Gst.TIME_ARGS(buf.pts)
        #logger.debug("timestamp(buffer):%s queue(buffer):%s" % (timestamp,
        #             self.blk_q.qsize()))
        res, bmap = buf.map(Gst.MapFlags.READ)
        self.blk_q.put((timestamp, bmap.data))

        return Gst.FlowReturn.OK


GObject.type_register(StreamExtractor)
__gstelementfactory__ = ("streamextractor", Gst.Rank.NONE, StreamExtractor)

