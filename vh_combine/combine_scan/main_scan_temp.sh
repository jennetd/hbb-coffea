#!/bin/bash

# Create a list of ddb and ddc score
declare -a pairs=(THRES_LIST)

eval `scramv1 runtime -sh`

for thres in "${pairs[@]}"
do
    IFS=" " read -r -a arr <<< "${thres}"

    i="${arr[0]}"
    j="${arr[1]}"

    echo "DDB: $i; DDC: $j"

    cd DDB-$i-DDC-$j/2017

    SignalRootFile=signalregion.root
    if [ -f "$SignalRootFile" ]; then
    echo "$SignalRootFile exists."
    else 
    hadd signalregion.root 1mvc-signalregion.root 1mvl-signalregion.root
    fi

    cd ..

    ln -sf ../make_cards.py .

    #Run make-cards
    python make_cards.py 2017

    #Run combine
    cd 2017
    ln -sf ../../make_workspace.sh .
    ln -sf ../../exp_significance.sh .

    pwd
    ls 

    source make_workspace.sh
    source exp_significance.sh

    cd ../../

    pwd
done
