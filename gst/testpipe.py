"""Module containing Pipeliner class."""
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

import sys
import yaml
import logging

from scribe.segment.detector import TurnDetector
from queue import Queue
from importlib import import_module

logger = logging.getLogger(__name__)


class Pipeliner(object):
    """Utility class to automate creation of GST pipeline."""

    def __init__(self, _plug, _cfg):
        """Initialize Pipeliner class."""
        # Gst loop
        self.gst_loop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline()
        self.custom = _plug
        self.plugins = {}

        # Turn detection loop
        self.blk_q = Queue()
        self.sentinel = object()
        self.td = TurnDetector()

        # Load Gst launch config and build pipeline
        try:
            cfg = yaml.load(open(_cfg))
            logger.debug("Creating test pipeline using conf: %s" % cfg)
            logger.debug("Test plugin is %s" % self.custom)
            self.prep(cfg)
        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))

    def prep(self, cfg):
        """Prepare Gst pipeline from specified yaml config file."""
        prev_element = None
        for plugin, props in cfg.items():
            # make element if non-test plugin
            if plugin != self.custom:
                element = Gst.ElementFactory.make(props['name'], plugin)
            else:
                plugin_module = import_module(
                    'scribe.gst.python.' + props['name'])
                element = plugin_module.GstPlugin(self.blk_q)
                logger.debug("Adding %s to the pipeline" % self.custom)
                self.element = element

            # set properties
            for k, v in props.items():
                # add exception for pesky caps and set properties
                if k == 'caps':
                    element.set_property(k, Gst.caps_from_string(v))
                elif k == 'name':
                    continue
                else:
                    element.set_property(k, v)

            # add elements before linking them
            self.plugins[plugin] = element
            self.pipeline.add(element)
            logger.debug("Adding %s to the pipeline with name %s" %
                    (props['name'], element.get_property('name')))

        for plugin, element in self.plugins.items():
            if prev_element:
                logger.debug("Linking %s to %s" % (prev_plugin, plugin))
                # link differently for decodebin
                if prev_plugin == 'decoder':
                    prev_element.connect('pad-added', self._connect_decoder)
                else:
                    prev_element.link(element)
            prev_plugin, prev_element = plugin, element

        # setup signal handling
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.enable_sync_message_emission()
        self.bus.connect('message::eos', self._on_eos)
        self.bus.connect('message::error', self._on_error)

        # set pipeline to ready
        self.pipeline.set_state(Gst.State.READY)
        logger.info("Pipeline READY")

    def _connect_decoder(self, element, pad):
        pad.link(self.plugins['converter1'].get_static_pad("sink"))
        logger.info("Connected audio decoder")

    def _on_eos(self, _bus, _msg):
        self.plugins['sink'].set_state(Gst.State.NULL)
        self.plugins['sink'].set_state(Gst.State.PLAYING)
        self.pipeline.set_state(Gst.State.NULL)

    def _on_error(self, _bus, _msg):
        err, dbg = _msg.parse_error()
        logger.debug("%s: %s" % (_msg.src.get_name(), err.message))
        self.on_eos(_bus, _msg)

    def play(self):
        self.pipeline.set_state(Gst.State.PAUSED)
        self.plugins['sink'].set_state(Gst.State.PLAYING)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            self.gst_loop.run()
        except KeyboardInterrupt:
            self.pipeline.set_state(Gst.State.NULL)
            sys.exit(0)
        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))
        finally:
            self.gst_loop.quit()
