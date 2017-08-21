#!/bin/bash

seg="$1.seg"
wav="$1.wav"

gmmInit="$1.init.gmm"
gmm="$1.gmm"

java -cp ~/down/prog/lium_spkdiarization-8.4.1.jar \
    fr.lium.spkDiarization.programs.MTrainInit \
    --sInputMask=$seg --fInputMask=$wav \
    --kind=FULL --nbComp=8 --emInitMethod=split_all --emCtrl=1,5,0.05 \
    --tOutputMask=$gmmInit "$1"

java -cp ~/down/prog/lium_spkdiarization-8.4.1.jar \
    fr.lium.spkDiarization.programs.MTrainEM \
    --sInputMask=$seg --fInputMask=$wav \
    --emCtrl=1,20,0.01 \
    --tInputMask=$gmmInit --tOutputMask=$gmm "$1"
