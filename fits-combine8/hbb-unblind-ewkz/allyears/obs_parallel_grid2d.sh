year=$1
npoints=2500

combineTool.py output/testModel${year}/model_combined.root -M MultiDimFit --algo grid --points ${npoints} --job-mode condor --split-points 10 --task-name '2dscan' -n '2dscan'
