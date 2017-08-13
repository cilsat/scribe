#!/bin/bash

wav=$1
seg=$2
gmm=$3
out=$4
ubm=$5
lium=$6

java -cp $lium fr.lium.spkDiarization.programs.Identification --help \
    --sInputMask=$seg --fInputMask=$wav --sOutputMask=$out \
    --fInputDesc="audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4" \
    --tInputMask=$gmm --sTop=5,$ubm --sSetLabel=add ${wav/.wav//}
