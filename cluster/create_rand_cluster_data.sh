#!/bin/zsh

# creates random concatenated audio files from randomnly chosen speakers
# returns a CSV of the durations and names of the files concatenated.

source path.sh

# usage:
root=$1   # path to audio dir containing speaker subdirs
num=$2    # number of speakers/files to take
model=$3  # path to model dir; needs .mdl, .fst
out=$4    # output path

# prepare out dir
[ ! -d "$out" ] && mkdir -p $out

# randomly choose speakers from root dir
spkr=($(ls $root | shuf | head -n $num))

# randomnly choose 1 to $num files from speaker dirs
files=($(for n in `seq 1 $num`; do \
  find $root/$spkr[$n] -iname "*.wav" | shuf | head -n $n; \
done | shuf))

# make scp file
fileids=()
for f in $files
do
  fid=${$(echo $f | rev | cut -d '/' -f1 | rev)%.wav}
  fileids+=($fid)
  echo $fid $f
done > $out.scp

# make transcript of files
echo $fileids
for f in $fileids
do
  grep $f $root/../text | cut -d ' ' -f2-
done > $out.txt

# concatenate chosen audio files and output to WAV
sox $files $out.wav

# compute raw MFCCs of concatenated audio file and output to MFC file
compute-mfcc-feats scp:$out.scp ark:- | \
  add-deltas ark:- ark:- | \
  copy-feats ark:- ark,t:$out.mfc

# generate lattice and align phones
gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 \
  --acoustic-scale=0.083333 --allow-partial=true \
  $model/final.mdl $model/HCLG.fst ark,t:$out.mfc ark:- | \
  lattice-align-phones --replace-output-symbols=true \
  $model/final.mdl ark:- ark:- | \
  lattice-1best ark:- ark:- | nbest-to-ctm ark:- $out.ctm

