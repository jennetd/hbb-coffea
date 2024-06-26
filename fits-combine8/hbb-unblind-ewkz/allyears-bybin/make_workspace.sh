year=""

if [[ "$PWD" == *"2016"* ]]; then
    year="_2016"
elif [[ "$PWD" == *"2017"* ]]; then
year="_2017"
elif [[ "$PWD" == *"2018"* ]]; then
    year="_2018"
fi

cd output/testModel${year}/

. build.sh

text2workspace.py model_combined.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
--PO 'map=ptbin0vbflo.*/VBF:rVBF7[1,-19,20]' --PO 'map=ptbin0vbfhi.*/VBF:rVBF8[1,-19,20]' \
--PO 'map=ptbin.*ggf.*/VBF:rVBF1[1,-50,100]' \
--PO 'map=ptbin0ggf.*/ggF:rggF1[1,-19,20]' --PO 'map=ptbin1ggf.*/ggF:rggF2[1,-19,20]' \
--PO 'map=ptbin2ggf.*/ggF:rggF3[1,-19,20]' --PO 'map=ptbin3ggf.*/ggF:rggF4[1,-19,20]' \
--PO 'map=ptbin4ggf.*/ggF:rggF5[1,-19,20]' --PO 'map=ptbin5ggf.*/ggF:rggF6[1,-19,20]' \
--PO 'map=ptbin0.*vbf.*/ggF:rggF7[1,-50,50]' 
