#!/usr/bin/env python

from scribe.gst.testpipe import Pipeliner
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pipeline = Pipeliner('sid', 'test.yaml')
    pipeline.play()
