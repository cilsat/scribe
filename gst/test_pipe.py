#!/usr/bin/env python

import sys
import json
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from splitsilence import SplitSilence

Gst.init(None)
loop = GLib.MainLoop()

src = Gst.ElementFactory.make('filesrc', 'filesrc')
wp = Gst.ElementFactory.make('wavparse', 'wavparse')
ss = SplitSilence()
sink = Gst.ElementFactory.make('autoaudiosink', 'autoaudiosink')

src.set_property('location', sys.argv[1])

pipeline = Gst.Pipeline()
for n in [src, wp, ss, sink]:
    pipeline.add(n)

src.link(wp)
wp.link(ss)
ss.link(sink)


def on_eos(_bus, _msg):
    ss.quit()
    loop.quit()

def on_error(_bus, _msg):
    err, dbg = _msg.parse_error()
    print(msg.src.get_name(), err.message)


bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect('message::eos', on_eos)
bus.connect('message::error', on_error)

pipeline.set_state(Gst.State.PLAYING)

loop.run()

pipeline.set_state(Gst.State.NULL)
