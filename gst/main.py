#!/usr/bin/env python

from scribe.gst.testpipe import Pipeliner
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = Pipeliner('pyextractor', 'test.yaml')
    pipeline.play()
