#!/bin/bash

source ./path.sh
utt=$1
model=$2
scp=${utt/wav/scp}

echo "$utt $utt" > $scp
compute-mfcc-feats scp:$scp ark:- | add-deltas ark:- ark:- | \
gmm-latgen-faster-parallel --num-threads=4 --allow-partial \
--word-symbol-table=$model/words.txt $model/final.mdl \
$model/HCLG.fst ark:- ark:/dev/null ark:/dev/null

