year="_2016"

modelfile=output/testModel${year}/model_combined.root

for bias in 0 1 `seq 5 5 25`
    do
    combineTool.py -M FitDiagnostics --setParameters rVBF=$bias,rggF=1 --trackParameters rggF,rVBF --trackErrors rggF,rVBF -n bias${bias}VBF -d ${modelfile} --cminDefaultMinimizerStrategy 0 --robustFit=1 -t 10 -s 1:100:1 --job-mode condor --task-name VBF$bias
    done
