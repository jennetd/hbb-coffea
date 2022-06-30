#!/bin/bash

# Create a list of ddb and ddc score
declare -a pairs=(THRES_LIST)

for thres in "${pairs[@]}"
do
   IFS=" " read -r -a arr <<< "${thres}"

   i="${arr[0]}"
   j="${arr[1]}"

   echo "DDB: $i; DDC: $j"
   
    #Make the workspace directory
    mkdir -p DDB-$i-DDC-$j/2017/

    #Copy all the stuff needed over
    cd DDB-$i-DDC-$j/2017/
    ln -sf ../../templates.pkl .
    cd ../../

    cp make-hists-1mv-charm.py DDB-$i-DDC-$j/
    cp make-hists-1mv-light.py DDB-$i-DDC-$j/

    cd DDB-$i-DDC-$j/

    ln -sf ../*.json .
    
    #Replace the threshold in make-hist
    sed -i "s/ddbthr =.*/ddbthr = $i/g" make-hists-1mv-*.py
    sed -i "s/ddcthr =.*/ddcthr = $j/g" make-hists-1mv-*.py

    #Run make-hist
    python make-hists-1mv-charm.py 2017
    python make-hists-1mv-light.py 2017

    cd ../

done

