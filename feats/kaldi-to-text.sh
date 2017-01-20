#!/bin/bash

source path.sh

if [[ $# -ne 4 ]]; then
  export mfcc_path="mfcc/spontan/test-spontan"
  export lat_path="exp/read-spontan/tri"
  export dec_path="decode-spontan"
  export out_path="exp/classifier/spontan"
else
  export mfcc_path=$1
  export lat_path=$2
  export dec_path=$3
  export out_path=$4
fi

# get raw mfcc frames for given test folder
for n in `ls $mfcc_path`
do
  for i in `ls $mfcc_path/$n/data/*mfcc*ark`
  do
    set=${i%.ark}
    add-deltas ark:$i ark:- | \
    copy-feats ark:- ark,t:$set.mfc
  done && \
  cat $mfcc_path/$n/data/*.mfc > $out_path/mfcc-$n.ali &
done

# get best path phone alignment from lattice in given folder
for n in `ls $lat_path`
do
  for i in `ls $lat_path/$n/$dec_path/lat*gz`
  do
    set=${i%.gz}
    lattice-align-phones --replace-output-symbols=true \
      $lat_path/$n/final.mdl \
      ark:"gunzip -c $i|" \
      ark:$set.lat && \
    lattice-1best ark:$set.lat \
      ark:- | nbest-to-ctm \
      ark:- $set.ctm
  done && \
  cat $lat_path/$n/$dec_path/*.ctm >> $out_path/phon-$n.ali &
done
