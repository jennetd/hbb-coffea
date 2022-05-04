# Determine the year from the directory name
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

text2workspace.py model_combined.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO 'map=.*/ZH:rVH[1,-9,10]' --PO verbose --PO 'map=.*/WH:rVH[1,-9,10]'

#--PO redefining the signal anyways, 
# map=.*/VBF: evaluate VBF in every category. 
# rVBF[1,-9,10]: name of para of interest, signal strength of vbf, [initial, min, max]
# --PO 'map=.*/ZH:rVH[1,-9,10]' --PO verbose --PO 'map=.*/WH:rVH[1,-9,10]' same scaling factor applied to both.