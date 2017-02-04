#!/bin/zsh

# creates random concatenated audio files from speech in a given directory.
# returns a CSV of the durations and names of the files concatenated.

source path.sh

root=$1
num=$2
model=$3
out=$4

# randomly choose audio files from root directory
spkr=($(ls $root | shuf | head -n $num))
files=()
for s in $spkr
do
  f=$(find $root/$s -iname "*.wav" | shuf | head -n 1)
  files+=($f)
  fn=$(echo $f | rev | cut -d '/' -f1 | rev)
  echo "${fn%.wav} $f"
done > $out.scp

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

