# Arguments                                                                     
year=$1
poi="rZbb"
frozen="allConstrainedNuisances"

npoints=1000

combineTool.py output/testModel/model_combined.root -M MultiDimFit --algo grid --points ${npoints} --redefineSignalPOI ${poi} --job-mode condor --split-points 10 --task-name ${poi} -n ${poi}

combineTool.py output/testModel/higgsCombineTest.MultiDimFit.mH125.root -M MultiDimFit --algo grid --points ${npoints} --redefineSignalPOI ${poi} -w w --snapshotName "MultiDimFit" --freezeParameters ${frozen} --job-mode condor --split-points 10 --task-name ${poi}Stat -n ${poi}'Stat'

