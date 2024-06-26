#!/bin/bash                                                                                                                                   

# Arguments
year=""
if [[ "$PWD" == *"2016APV"* ]]; then
    year="_2016APV"
elif [[ "$PWD" == *"2016"* ]]; then
    year="_2016"
elif [[ "$PWD" == *"2017"* ]]; then
    year="_2017"
elif [[ "$PWD" == *"2018"* ]]; then
    year="_2018"
fi

modelfile=output/testModel${year}/model_combined.root

# Collect results into json
combineTool.py -M Impacts -d $modelfile -m 125 -o impacts.json

poiarray=('rggF300to450' 'rggF450to650' 'rggF650plus' 'rVBF1500plus' 'rVBF1000to1500')

for poi in ${poiarray[@]}; do
    echo $poi
    plotImpacts.py -i impacts.json -o impacts_${poi} --POI $poi --blind
done
