#!/usr/bin/env python

import sys
import json
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from collections import OrderedDict

Gst.init(None)


class Pipeliner(object):

    def __init__(self, config):
        self.loop = GLib.MainLoop()

        self.plugins = [Gst.ElementFactory.make(
            plug, plug) for plug in config.keys()]

        self.pipeline = Gst.Pipeline()
        for n, plug in enumerate(config.keys()):
            print(plug)
            options = config[plug]
            if options:
                for o in options.keys():
                    try:
                        self.plugins[n].set_property(o, config[plug][o])
                    except Exception as e:
                        print(e)
            self.pipeline.add(self.plugins[n])

        for n in range(len(config.keys()) - 1):
            self.plugins[n].link(self.plugins[n + 1])

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::element', self.on_cut)
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)

    def run(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Error opening pipeline")
            sys.exit(1)

        self.loop.run()

    def on_cut(self, bus, msg):
        s = msg.get_structure()
        lvl = 'below'
        if s['above']:
            lbl = 'above'
        print(Gst.TIME_ARGS(s['timestamp']), lvl)

    def on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        print("Error:", msg.src.get_name(), ":", err.message)
        if dbg:
            print("Debug info:", dbg)

    def on_eos(self, bus, msg):
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()


if __name__ == '__main__':
    with open(sys.argv[1]) as cfg:
        config = json.load(cfg, object_pairs_hook=OrderedDict)

    p = Pipeliner(config)
    p.run()
