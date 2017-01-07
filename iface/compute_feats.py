#!/usr/bin/python3
# Requires multiprocessing with starmap (only available in python 3) and Pandas

from iface import parse_files
import feats

import os
from argparse import ArgumentParser

# process all alignment files in a given directory
# MFCC files are assumed to begin with 'mfcc' and phone alignments files with
# 'phon'. returns dataframe indexed by utterance name containing mfccs,
# delta/delta2s, and phone order within utterance
def process_path(path, output='feats.hdf'):
    outpath = os.path.join(path, output)

    print("parsing files")
    mfiles = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('mfcc')]
    pfiles = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('phon')]
    mfcc, phon = parse_files(mfiles, pfiles)
    print(mfcc.info())
    print(phon.info())
    print("\nwriting MFCC frames and phone alignments to hdf")
    mfcc.to_hdf(outpath, 'mfcc')
    phon.to_hdf(outpath, 'phon')

    print("\ncomputing deltas")
    delta = feats.compute_deltas(mfcc)
    del mfcc
    print(delta.info())
    print("\nwriting full MFCCs to hdf")
    delta.to_hdf(outpath, 'delta')

    print("\naligning MFCC frames to phones")
    ali = feats.align_phones(delta, phon)
    del delta, phon
    print(ali.info())
    print("\n writing aligned frames to hdf")
    ali.to_hdf(outpath, 'ali')

    print("\ncomputing segment level features")
    seg = feats.compute_seg_feats(ali)
    print(seg.info())
    print("\nwriting segment features to hdf")
    seg.to_hdf(outpath, 'seg')
    del ali

    print("\ncomputing utterance level features")
    utt = feats.compute_utt_feats(seg)
    print(utt.info())
    print("\nwriting utterance features to hdf")
    utt.to_hdf(outpath, 'utt')
    del seg

    print("\nfinished processing " + output)


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute frame, segment, and utterance \
            level features from a set of Kaldi phone alignment (CTM) files and \
            MFCCs file, obtainable from the kaldi-to-text.sh script.")
    parser.add_argument('path', type=str,
        help="Location of alignment files; phone alignment files must have the \
                'phon' prefix and MFCC alignment files must have the 'mfcc-' \
                prefix.")
    parser.add_argument('file', help="Location of output HDF file to store \
            features in.")

    args = parser.parse_args()

    process_path(args.path, args.file)

