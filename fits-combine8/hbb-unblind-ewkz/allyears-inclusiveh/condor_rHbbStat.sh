#!/bin/sh
ulimit -s unlimited
set -e

xrdcp -s root://cmseos.fnal.gov//store/user/jennetd/CMSSW_10_2_13.tar.gz .
source /cvmfs/cms.cern.ch/cmsset_default.sh
tar -xf CMSSW_10_2_13.tar.gz
rm CMSSW_10_2_13.tar.gz
cd CMSSW_10_2_13/src/
scramv1 b ProjectRename
eval `scramv1 runtime -sh`
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ${_CONDOR_SCRATCH_DIR}
ls


if [ $1 -eq 0 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 0 --lastPoint 9 -n rHbbStat.POINTS.0.9
fi
if [ $1 -eq 1 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 10 --lastPoint 19 -n rHbbStat.POINTS.10.19
fi
if [ $1 -eq 2 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 20 --lastPoint 29 -n rHbbStat.POINTS.20.29
fi
if [ $1 -eq 3 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 30 --lastPoint 39 -n rHbbStat.POINTS.30.39
fi
if [ $1 -eq 4 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 40 --lastPoint 49 -n rHbbStat.POINTS.40.49
fi
if [ $1 -eq 5 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 50 --lastPoint 59 -n rHbbStat.POINTS.50.59
fi
if [ $1 -eq 6 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 60 --lastPoint 69 -n rHbbStat.POINTS.60.69
fi
if [ $1 -eq 7 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 70 --lastPoint 79 -n rHbbStat.POINTS.70.79
fi
if [ $1 -eq 8 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 80 --lastPoint 89 -n rHbbStat.POINTS.80.89
fi
if [ $1 -eq 9 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 100 --firstPoint 90 --lastPoint 99 -n rHbbStat.POINTS.90.99
fi