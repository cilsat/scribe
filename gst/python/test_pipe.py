#!/usr/bin/env python

import sys
import json
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from splitsilence import SplitSilence

Gst.init(None)
loop = GLib.MainLoop()

ss = SplitSilence()
pipeline = Gst.Pipeline()


def test_stream():
    src = Gst.ElementFactory.make('pulsesrc', 'pulsesrc')
    ac = Gst.ElementFactory.make('audioconvert', 'audioconvert')
    ar = Gst.ElementFactory.make('audioresample', 'audioresample')
    caps = Gst.ElementFactory.make('capsfilter',
                                   'audio/x-raw,format=S16LE,rate=16000,channels=1')
    sink = Gst.ElementFactory.make('autoaudiosink', 'autoaudiosink')

    caps.set_property('caps', Gst.caps_from_string(
        'audio/x-raw, rate=16000, width=16, depth=16, channels=1'))

    for n in [src, ac, ar, caps, ss, sink]:
        pipeline.add(n)

    src.link(ac)
    ac.link(ar)
    ac.link(ar)
    ar.link(caps)
    caps.link(ss)
    ss.link(sink)


def test_file():
    src = Gst.ElementFactory.make('filesrc', 'filesrc')
    src.set_property('location', sys.argv[2])
    wp = Gst.ElementFactory.make('wavparse', 'wavparse')
    sink = Gst.ElementFactory.make('autoaudiosink', 'autoaudiosink')

    for n in [src, wp, ss, sink]:
        pipeline.add(n)

    src.link(wp)
    wp.link(ss)
    ss.link(sink)


def on_eos(_bus, _msg):
    pipeline.set_state(Gst.State.NULL)
    ss.quit()
    loop.quit()


def on_error(_bus, _msg):
    err, dbg = _msg.parse_error()
    print(_msg.src.get_name(), err.message)


def play():
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message::eos', on_eos)
    bus.connect('message::error', on_error)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pipeline.set_state(Gst.State.NULL)
        sys.exit(0)
    except Exception as e:
        sys.exit(type(e).__name__ + ': ' + str(e))
    finally:
        ss.quit()
        loop.quit()


if __name__ == '__main__':
    if sys.argv[1] == 'stream':
        test_stream()
    elif sys.argv[1] == 'file':
        test_file()

    play()
