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
filenames=()
for s in $spkr
do
  f=$(find $root/$s -iname "*.wav" | shuf | head -n 1)
  files+=($f)
  fn=$(echo $f | rev | cut -d '/' -f1 | rev)
  fn=${fn%.wav}
  filenames+=($fn)
  echo "${fn%.wav} $f"
done > $out.scp

# make transcript of files
echo $filenames
for f in $filenames
do
  tr=$(grep $f $root/../text | cut -d ' ' -f2-)
  echo $tr
done > $out.txt

# concatenate chosen audio files and output to WAV
sox $files $out.wav

# compute raw MFCCs of concatenated audio file and output to MFC file
compute-mfcc-feats scp:$out.scp ark:- | \
  copy-feats ark:- ark,scp:$out.ark,$out"_feats.scp"
add-deltas scp:$out"_feats.scp" ark:- | \
  copy-feats ark:- ark,t:$out.mfc

# compute VAD of extracted feats
compute-vad scp:$out"_feats.scp" ark,t:$out.vad

# generate lattice and align phones
gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 \
  --acoustic-scale=0.083333 --allow-partial=true \
  $model/final.mdl $model/HCLG.fst ark,t:$out.mfc ark:- | \
  lattice-align-phones --replace-output-symbols=true \
  $model/final.mdl ark:- ark:- | \
  lattice-1best ark:- ark:- | nbest-to-ctm ark:- $out.ctm

