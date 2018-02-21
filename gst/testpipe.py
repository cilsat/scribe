import sys
import gi
import yaml
import logging

from importlib import import_module
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

logger = logging.getLogger(__name__)


class Pipeliner:
    """
    Custom class to initialize and run Gst pipeline for custom python Gst
    plugin given a YAML config file.
    """

    def __init__(self, _plug, _cfg):
        self.loop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline()
        self.plugin = _plug

        try:
            cfg = yaml.load(open(_cfg))
            logger.debug("Creating test pipeline using conf: %s" % cfg)
            logger.debug("Test plugin is %s" % self.plugin)
            self.prep(cfg)
        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::eos', self.on_eos)
        bus.connect('message::error', self.on_error)

        ret = self.pipeline.set_state(Gst.State.READY)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.debug("Error readying pipeline")
            sys.exit(0)
        else:
            logger.debug("Pipeline READY")

    def prep(self, cfg):
        """
        Prepare Gst pipeline from specified yaml config file.
        """
        prev_element = None
        for plugin, props in cfg.items():
            # make element if non-test plugin
            if plugin != self.plugin:
                element = Gst.ElementFactory.make(plugin)
                logger.debug("Adding %s to the pipeline" % plugin)
                if plugin == 'onlinegmmdecodefaster':
                    self.decoder = element
            else:
                plugin_module = import_module(
                    'scribe.gst.python.' + self.plugin)
                element = plugin_module.GstPlugin()
                logger.debug("Adding %s to the pipeline" % self.plugin)
                self.element = element
            for k, v in props.items():
                logger.debug("Setting %s with value %s" % (k, v))
                # add exception for pesky caps and set properties
                if k == 'caps':
                    element.set_property(k, Gst.caps_from_string(v))
                else:
                    element.set_property(k, v)
            # add element to pipeline and link previous element to current one
            self.pipeline.add(element)
            if prev_element:
                prev_element.link(element)
            prev_element = element

    def on_eos(self, _bus, _msg):
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()

    def on_error(self, _bus, _msg):
        err, dbg = _msg.parse_error()
        logger.debug("%s: %s" % (_msg.src.get_name(), err.message))
        self.on_eos(_bus, _msg)

    def play(self):
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.debug("Error opening pipeline")
            sys.exit(0)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.pipeline.set_state(Gst.State.NULL)
            sys.exit(0)
        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))
        finally:
            self.loop.quit()
