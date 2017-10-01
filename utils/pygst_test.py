#!/usr/bin/env python2

import sys
import gi
gi.require_version('Gst', '1.0')

from gi.repository import Gst, GLib

Gst.init(None)


class Cutter(object):

    def __init__(self):
        self.loop = GLib.MainLoop()

        self.src = Gst.ElementFactory.make('filesrc', 'filesrc')
        self.wavparse = Gst.ElementFactory.make('wavparse', 'wavparse')
        self.cutter = Gst.ElementFactory.make('cutter', 'cutter')
        self.sink = Gst.ElementFactory.make('autoaudiosink', 'autoaudiosink')

        self.src.set_property('location', sys.argv[1])
        #self.cutter.set_property('threshold-dB', -5.0)

        self.pipeline = Gst.Pipeline()
        for element in [self.src, self.wavparse, self.cutter, self.sink]:
            self.pipeline.add(element)

        self.src.link(self.wavparse)
        self.wavparse.link(self.cutter)
        self.cutter.link(self.sink)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::element', self.on_cut)
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)

    def start(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR")
            sys.exit(1)

        self.loop.run()

        self.pipeline.set_state(Gst.State.NULL)

    def on_cut(self, bus, msg):
        print(msg.get_structure().get_name())
        #s = msg.structure
        #lvl = 'below'
        # if s['above']:
        #    lbl = 'above'
        #print(Gst.TIME_ARGS(s['timestamp']), lvl)

    def on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        print("Error:", msg.src.get_name(), ":", err.message)
        if dbg:
            print("Debug info:", dbg)

    def on_eos(self, bus, msg):
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()


if __name__ == '__main__':
    c = Cutter()
    c.start()
