year=""

cd output/testModel${year}/

. build.sh

text2workspace.py model_combined.txt -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO 'map=.*/VBF:rVBF[1,-10,20]' --PO 'map=.*/ggF:rggF[1,-10,20]' --PO verbose 

