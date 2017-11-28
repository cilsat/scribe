#!/usr/bin/bash

thr=("1.5:2.5,2.5:3.5,250.0:300,0:3.0" "1.0:2.5,2.5:3.5,250.0:300,0:3.0" "0.0:2.5,2.5:3.5,250.0:300,0:3.0" "1.5:2.5,3.0:3.5,250.0:300,0:3.0" "1.5:2.5,3.5:4.5,250.0:300,0:3.0" "0.0:2.5,3.5:4.5,250.0:300,0:3.0" "0.0:2.5,3.5:5.0,250.0:300,0:3.0" "0.0:2.5,4.0:5.0,250.0:300,0:3.0")

for t in $thr
do
    for file in $(ls $1/*.wav)
    do
        java -Xmx4096m -jar ~/net/Files/lium_spkdiarization-8.4.1.jar \
            --thresholds=$t --fInputMask="$file" --sInputMask="${file/.wav/.uem.seg}"
            --doCEClustering ${file/.wav} &> ${file/.wav/.txt}
    done
done


