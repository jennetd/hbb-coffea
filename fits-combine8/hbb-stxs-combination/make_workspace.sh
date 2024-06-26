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
		  --PO 'map=.*/ggH_PTH_300_450_PTHJoverPTH_0_15:rggF300to450[1,-70,70]' \
		  --PO 'map=.*/ggH_PTH_300_450_PTHJoverPTH_GT15:rggF300to450[1,-70,70]' \
		  --PO 'map=.*/ggH_PTH_450_650_PTHJoverPTH_0_15:rggF450to650[1,-70,70]' \
		  --PO 'map=.*/ggH_PTH_450_650_PTHJoverPTH_GT15:rggF450to650[1,-70,70]' \
		  --PO 'map=.*/ggH_PTH_GT650_PTHJoverPTH_0_15:rggF650plus[1,-70,70]' \
		  --PO 'map=.*/ggH_PTH_GT650_PTHJoverPTH_GT15:rggF650plus[1,-70,70]' \
		  --PO 'map=.*/qqH_GE2J_MJJ_1000_1500_PTH_GT200_PTHJJ_0_25:rVBF1000to1500[1,-70,70]' \
		  --PO 'map=.*/qqH_GE2J_MJJ_1000_1500_PTH_GT200_PTHJJ_GT25:rVBF1000to1500[1,-70,70]' \
		  --PO 'map=.*/qqH_GE2J_MJJ_GT1500_PTH_GT200_PTHJJ_0_25:rVBF1500plus[1,-70,70]' \
		  --PO 'map=.*/qqH_GE2J_MJJ_GT1500_PTH_GT200_PTHJJ_GT25:rVBF1500plus[1,-70,70]'
