year=""

if [[ "$PWD" == *"2016APV"* ]]; then
    year="_2016APV"
elif [[ "$PWD" == *"2016"* ]]; then
    year="_2016"
elif [[ "$PWD" == *"2017"* ]]; then
    year="_2017"
elif [[ "$PWD" == *"2018"* ]]; then
    year="_2018"
elif [[ "$PWD" == *"2015"* ]]; then
    year="_2015"
fi

cd output/testModel${year}/

. build.sh

text2workspace.py model_combined.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO 'map=.*/ggF_s1:rggF300to450[1,-70,70]' \
    --PO 'map=.*/ggF_s2:rggF450to650[1,-70,70]' \
    --PO 'map=.*/ggF_s3:rggF650plus[1,-70,70]' \
    --PO 'map=.*/VBF_s1:rVBF1000to1500[1,-70,70]' \
    --PO 'map=.*/VBF_s2:rVBF1500plus[1,-70,70]'

