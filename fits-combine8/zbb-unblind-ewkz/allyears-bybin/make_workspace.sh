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
		  --PO 'map=ptbin0vbflo.*/Zjetsbb:rZbb7[1,-19,20]' --PO 'map=ptbin0vbflo.*/EWKZbb:rZbb7[1,-19,20]' \
		  --PO 'map=ptbin0vbfhi.*/Zjetsbb:rZbb8[1,-19,20]' --PO 'map=ptbin0vbfhi.*/EWKZbb:rZbb8[1,-19,20]'\
		  --PO 'map=ptbin0ggf.*/Zjetsbb:rZbb1[1,-19,20]' --PO 'map=ptbin0ggf.*/EWKZbb:rZbb1[1,-19,20]'\
		  --PO 'map=ptbin1ggf.*/Zjetsbb:rZbb2[1,-19,20]' --PO 'map=ptbin1ggf.*/EWKZbb:rZbb2[1,-19,20]'\
		  --PO 'map=ptbin2ggf.*/Zjetsbb:rZbb3[1,-19,20]' --PO 'map=ptbin2ggf.*/EWKZbb:rZbb3[1,-19,20]'\
		  --PO 'map=ptbin3ggf.*/Zjetsbb:rZbb4[1,-19,20]' --PO 'map=ptbin3ggf.*/EWKZbb:rZbb4[1,-19,20]'\
		  --PO 'map=ptbin4ggf.*/Zjetsbb:rZbb5[1,-19,20]' --PO 'map=ptbin4ggf.*/EWKZbb:rZbb5[1,-19,20]'\
		  --PO 'map=ptbin5ggf.*/Zjetsbb:rZbb6[1,-19,20]' --PO 'map=ptbin5ggf.*/EWKZbb:rZbb6[1,-19,20]'
