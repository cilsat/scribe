#!/usr/bin/env python

from testpipe import Pipeliner
from python.splitsilence import GstPlugin
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pipeline = Pipeliner('splitsilence', 'test.yaml')
    pipeline.play()
