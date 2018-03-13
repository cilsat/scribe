#!/usr/bin/env python
"""Gst app for demo."""

import os
import sys
import time
import logging
import gbulb
import asyncio
import uvloop
import json

import numpy as np
import pandas as pd
import soundfile as sf
import sounddevice as sd

from subprocess import run
from time import sleep, perf_counter
from queue import Queue
from importlib import import_module
from threading import Thread
from argparse import ArgumentParser
from tempfile import mkstemp

from scribe.segment.detector import TurnDetector

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
Gst.init(None)
gbulb.install()

logger = logging.getLogger(__name__)


class DemoPipe(object):
    def __init__(self, mdl, fst, txt, on_word=None):
        self.pipeline = Gst.Pipeline()
        self._on_word = on_word or self.on_word

        plugins = {
            'source': 'appsrc',
            'decode': 'decodebin',
            'convert': 'audioconvert',
            'resample': 'audioresample',
            'asr': 'onlinegmmdecodefaster',
            'sink': 'fakesink'
        }

        self.elements = {}
        for e, n in plugins.items():
            element = Gst.ElementFactory.make(n, e)
            if n == 'onlinegmmdecodefaster':
                element.set_property('model', mdl)
                element.set_property('fst', fst)
                element.set_property('word-syms', txt)
                element.set_property('silence-phones', '6:39:43')
            elif n == 'filesink':
                element.set_property('location', '/dev/stdout')
            elif n == 'appsrc':
                # caps = 'audio/x-raw,format=S16LE,rate=16000,channels=1'
                element.set_property('caps', None)
                element.set_property('is-live', True)
            self.elements[e] = element
            self.pipeline.add(element)
            logger.debug("Element %s added to pipeline as %s" % (n, e))

        self.elements['source'].link(self.elements['decode'])
        self.elements['convert'].link(self.elements['resample'])
        self.elements['resample'].link(self.elements['asr'])
        self.elements['asr'].link(self.elements['sink'])

        # Setup signal handling
        self.elements['decode'].connect('pad-added', self.connect_decoder)
        self.elements['asr'].connect('hyp-word', self._on_word)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::eos', self.on_eos)
        self.bus.connect('message::error', self.on_error)

        # Set pipeline to ready
        self.pipeline.set_state(Gst.State.READY)
        logger.info("Pipeline READY")

        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        buf = Gst.Buffer.new_allocate(None, 0, None)
        self.elements['source'].emit("push-buffer", buf)
        logger.info('Pipeline initialized')

    def connect_decoder(self, element, pad):
        pad.link(self.elements['convert'].get_static_pad("sink"))
        logger.debug("Connected audio decoder.")

    def on_word(self, asr, word):
        logger.info("Received word: %s" % word)

    async def process_data(self, data):
        # logger.debug('Pushing buffer of size %d to pipeline' % len(data))
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        self.elements['source'].emit("push-buffer", buf)

    def on_eos(self, _bus, _msg):
        logger.debug("EOS reached.")
        self.elements['source'].emit("end-of-stream")
        self.elements['sink'].set_state(Gst.State.NULL)
        self.pipeline.set_state(Gst.State.NULL)

    def on_error(self, _bus, _msg):
        err, dbg = _msg.parse_error()
        logger.debug("%s: %s" % (_msg.src.get_name(), err.message))
        self.on_eos(_bus, _msg)


class DemoApp(object):
    def __init__(self, loop, data, lium, kaldi):
        self.data = data
        self.samplerate = 16000

        # Validate kaldi dir
        self.mdl = os.path.join(kaldi, 'final.mdl')
        self.fst = os.path.join(kaldi, 'HCLG.fst')
        self.txt = os.path.join(kaldi, 'words.txt')

        assert os.path.isfile(self.mdl)
        assert os.path.isfile(self.fst)
        assert os.path.isfile(self.txt)

        # Validate LIUM dir
        self.jar = os.path.join(lium, 'lium-8.4.1.jar')
        self.wav = os.path.join(lium, 'spk.wav')
        self.seg = os.path.join(lium, 'spk.seg')
        self.ubm = os.path.join(lium, 'ubm.gmm')
        self.igmm = os.path.join(lium, 'spk.i.gmm')
        self.gmm = os.path.join(lium, 'spk.gmm')
        self.log = os.path.join(lium, 'spk.log')

        assert os.path.isfile(self.jar)
        assert os.path.isfile(self.ubm)
        assert os.path.isfile(self.gmm)

        self.data_queue = asyncio.Queue(loop=loop)
        self.words = []
        self.res = pd.DataFrame(columns=['time', 'word'])

        self.pipe = DemoPipe(self.mdl, self.fst, self.txt,
                             on_word=self.on_word)

        self.td = TurnDetector(lium=self.jar, ubm=self.ubm, block_size=8000,
                               gmm=self.gmm, energy_thr=4.5, sil_len_thr=0.25,
                               out_dir=self.data, on_result=self.on_speaker)

    def enroll_user(self):
        print("Nama: ")
        user = input()
        print("Jenis kelamin (M/F): ")
        gender = input()
        print("Baca kalimat-kalimat berikut dan tekan Ctrl+C setelahnya:")
        print("Seorang pejabat tinggi di kantor menteri negara BUMN " +
              "menyatakan, keputusan perpanjangan PKPS merupakan keputusan " +
              "yang sangat berat dan dilematis bagi pemerintah.")
        print("Musibah tanah longsor di Mojokerto, Jawa Timur baru baru ini " +
              "merupakan dampak nyata perusakan lingkungan yang disebabkan " +
              "oleh ulah manusia sendiri.")
        print("Hal itu sesuai dengan peraturan WTO bahwa sebuah negara yang " +
              "melakukan konsumsi atas bahan pokoknya sebesar lima persen " +
              "dari konsumsi nasional bisa menaikkan tarif sampai di atas " +
              "seratus persen.")

        rate = self.samplerate
        _, out_file = mkstemp(suffix='.wav', dir=self.data)
        try:
            q = Queue()

            def callback(data, frames, time, status):
                if status:
                    logger.debug("Status: %s" % status)
                q.put(data.copy())

            with sf.SoundFile(out_file, mode='w', samplerate=rate, channels=1,
                              subtype='PCM_16') as f:
                with sd.InputStream(samplerate=rate, channels=1,
                                    callback=callback):
                    print('Tekan Ctrl+C untuk menghentikan rekaman.')
                    while True:
                        f.write(q.get())
        except KeyboardInterrupt:
            print("Rekaman berhasil disimpan di %s." % out_file)
        except Exception as e:
            sys.exit(type(e).__name__ + ': ' + str(e))

        columns = ['wav', 'ch', 'start', 'dur', 'gen', 's', 'u', 'spk']
        if os.path.isfile(self.seg):
            self.segdf = pd.read_csv(self.seg, header=None,
                                     sep=' ', names=columns)
        else:
            self.segdf = pd.DataFrame(columns=columns)

        info = sf.info(out_file)
        if len(self.segdf) > 0:
            last = self.segdf.iloc[-1]
            start = last.start + last.dur
        else:
            start = 0
        new = {'wav': out_file, 'ch': 1, 'start': start,
               'dur': int(info.duration * 100), 'gen': gender, 's': 'S',
               'u': 'U', 'spk': user}

        self.segdf = self.segdf.append(new, ignore_index=True)
        self.segdf.to_csv(self.seg, sep=' ', header=None, index=False)
        self.sox_concat(self.segdf.wav.tolist(), self.wav, self.log)
        self.train(self.jar, self.seg, self.wav, self.ubm,
                   self.igmm, self.gmm, 'spk', self.log)

        print("Pendaftaran berhasil.")

    @staticmethod
    def sox_concat(infiles, outfile, log):
        cmd = ['sox'] + infiles + [outfile]
        with open(log, 'a') as f:
            run(cmd, stderr=f)

    @staticmethod
    def train(lium, seg, wav, ubm, igmm, gmm, name, log):
        # initialize speaker models using ubm
        init_cmd = [
            'java', '-cp', lium, 'fr.lium.spkDiarization.programs.MTrainInit',
            '--sInputMask=' + seg, '--fInputMask=' + wav,
            '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
            '--emInitMethod=copy', '--tInputMask=' + ubm,
            '--tOutputMask=' + igmm, name
        ]

        # train speaker models
        train_cmd = [
            'java', '-cp', lium, 'fr.lium.spkDiarization.programs.MTrainMAP',
            '--sInputMask=' + seg, '--fInputMask=' + wav,
            '--fInputDesc=audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4',
            '--tInputMask=' + igmm, '--emCtrl=1,5,0.01', '--varCtrl=0.01,10.0',
            '--tOutputMask=' + gmm, name
        ]

        with open(log, 'w') as f:
            run(init_cmd, stderr=f)
        with open(log, 'a') as f:
            run(train_cmd, stderr=f)

    def on_word(self, asr, word):
        self.words.append(word)

    def on_speaker(self, speaker):
        words = ' '.join([w for w in self.words if not w.startswith('<')])
        result = json.dumps({'sid': speaker, 'txt': words})
        logger.info('%s' % result)
        self.words = []

    async def device_loop(self):
        with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16',
                               blocksize=4000) as s:
            logger.info("Memulai pengenalan")
            while True:
                block = s.read(4000)[0][:]
                if block == b'':
                    break
                logger.debug("%d" % len(block))
                await self.data_queue.put(block)

    async def file_loop(self, filename):
        with open(filename, 'rb') as f:
            logger.info("Memulai pengenalan")
            # Loop
            while True:
                block = f.read(8000)
                if not block:
                    break
                await self.data_queue.put(block)

    async def data_loop(self, loop):
        call = 0
        while True:
            data = await self.data_queue.get()
            await self.pipe.process_data(data)
            if len(data) < 8000:
                loop.run_in_executor(None, self.td.process_block, data, True)
            else:
                loop.run_in_executor(None, self.td.process_block, data)
            wait = 0.25 - perf_counter() + call
            if wait >= 0:
                sleep(wait)
            call = perf_counter()
            if data is None:
                break
            self.data_queue.task_done()

    def file_play(self, filename):
        data, sr = sf.read(filename)
        sd.play(data, sr)

    async def decode(self, loop, file=None, online=False):
        asyncio.ensure_future(self.data_loop(loop))
        if online:
            asyncio.ensure_future(self.device_loop())
        else:
            loop.run_in_executor(None, self.file_play, file)
            await self.file_loop(file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser(description="Demo app for online speaker \
                            identification.")
    parser.add_argument('--data', type=str, default='/tmp', help="Default \
                        location for storing files.")
    parser.add_argument('--kaldi', type=str,
                        default='/home/cilsat/dev/prosa/scribe/kaldi',
                        help="Path to directory containing files needed for \
                        speech recognition with Kaldi.")
    parser.add_argument('--lium', type=str,
                        default="/home/cilsat/dev/prosa/scribe/lium",
                        help="Path to directory containing files needed for \
                        LIUM speaker identification.")
    parser.add_argument('--enroll', action='store_true', help="Run \
                        interactively for speaker enrollment.")
    parser.add_argument('--online', action='store_true', help="Run decoding \
                        session from recording device.")
    parser.add_argument('--file', type=str, help="Run decoding session on the \
                        given file path.", default=None)
    args = parser.parse_args()
    logger.debug("%s" % args)

    loop = asyncio.get_event_loop()
    app = DemoApp(loop, args.data, args.lium, args.kaldi)

    if args.enroll:
        app.enroll_user()
    else:
        try:
            if not args.online:
                assert os.path.isfile(args.file)
                info = sf.info(args.file)
                assert info.samplerate == 16000
                assert info.channels == 1
            loop.run_until_complete(app.decode(
                loop, file=args.file, online=args.online))
        except KeyboardInterrupt:
            logger.info("Pengenalan dihentikan.")
        finally:
            loop.close()
