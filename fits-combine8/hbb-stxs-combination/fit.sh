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

# Do initial fit                                                                                          
combine -M MultiDimFit --algo singles -d $modelfile -m 125 --setParameters rggF=1,rVBF=1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0


