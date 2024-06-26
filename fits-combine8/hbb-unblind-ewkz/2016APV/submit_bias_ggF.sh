year="_2016APV"

modelfile=output/testModel${year}/model_combined.root

for bias in 0 1 `seq 5 5 25`
    do
    combineTool.py -M FitDiagnostics --setParameters rVBF=1,rggF=$bias --trackParameters rggF,rVBF --trackErrors rggF,rVBF -n bias${bias}ggF -d ${modelfile} --cminDefaultMinimizerStrategy 0 --robustFit=1 -t 10 -s 1:100:1 --job-mode condor --task-name ggF$bias
    done
