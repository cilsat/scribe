#!/bin/bash

seg=$1
wav=$2
gmmInit=$3
gmm=$4
ubm=$5

lium="/home/cilsat/src/lium/out/artifacts/lium_jar/lium.jar"

#copy the ubm for each speaker
java -cp $lium fr.lium.spkDiarization.programs.MTrainInit \
    --sInputMask=$seg --fInputMask=$wav \
    --fInputDesc="audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4"  \
    --emInitMethod=copy --tInputMask=$ubm --tOutputMask=$gmmInit speakers
 
#train (MAP adaptation, mean only) of each speaker, the diarization file describes the training data of each speaker.
java -cp $lium fr.lium.spkDiarization.programs.MTrainMAP \
    --sInputMask=$seg --fInputMask=$wav \
    --fInputDesc="audio16kHz2sphinx,1:3:2:0:0:0,13,1:1:300:4"  \
    --tInputMask=$gmmInit --emCtrl=1,5,0.01 --varCtrl=0.01,10.0 \
    --tOutputMask=$gmm speakers
     
