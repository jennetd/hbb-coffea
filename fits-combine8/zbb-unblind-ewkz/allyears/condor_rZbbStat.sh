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
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 0 --lastPoint 9 -n rZbbStat.POINTS.0.9
fi
if [ $1 -eq 1 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 10 --lastPoint 19 -n rZbbStat.POINTS.10.19
fi
if [ $1 -eq 2 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 20 --lastPoint 29 -n rZbbStat.POINTS.20.29
fi
if [ $1 -eq 3 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 30 --lastPoint 39 -n rZbbStat.POINTS.30.39
fi
if [ $1 -eq 4 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 40 --lastPoint 49 -n rZbbStat.POINTS.40.49
fi
if [ $1 -eq 5 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 50 --lastPoint 59 -n rZbbStat.POINTS.50.59
fi
if [ $1 -eq 6 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 60 --lastPoint 69 -n rZbbStat.POINTS.60.69
fi
if [ $1 -eq 7 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 70 --lastPoint 79 -n rZbbStat.POINTS.70.79
fi
if [ $1 -eq 8 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 80 --lastPoint 89 -n rZbbStat.POINTS.80.89
fi
if [ $1 -eq 9 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 90 --lastPoint 99 -n rZbbStat.POINTS.90.99
fi
if [ $1 -eq 10 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 100 --lastPoint 109 -n rZbbStat.POINTS.100.109
fi
if [ $1 -eq 11 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 110 --lastPoint 119 -n rZbbStat.POINTS.110.119
fi
if [ $1 -eq 12 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 120 --lastPoint 129 -n rZbbStat.POINTS.120.129
fi
if [ $1 -eq 13 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 130 --lastPoint 139 -n rZbbStat.POINTS.130.139
fi
if [ $1 -eq 14 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 140 --lastPoint 149 -n rZbbStat.POINTS.140.149
fi
if [ $1 -eq 15 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 150 --lastPoint 159 -n rZbbStat.POINTS.150.159
fi
if [ $1 -eq 16 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 160 --lastPoint 169 -n rZbbStat.POINTS.160.169
fi
if [ $1 -eq 17 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 170 --lastPoint 179 -n rZbbStat.POINTS.170.179
fi
if [ $1 -eq 18 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 180 --lastPoint 189 -n rZbbStat.POINTS.180.189
fi
if [ $1 -eq 19 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 190 --lastPoint 199 -n rZbbStat.POINTS.190.199
fi
if [ $1 -eq 20 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 200 --lastPoint 209 -n rZbbStat.POINTS.200.209
fi
if [ $1 -eq 21 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 210 --lastPoint 219 -n rZbbStat.POINTS.210.219
fi
if [ $1 -eq 22 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 220 --lastPoint 229 -n rZbbStat.POINTS.220.229
fi
if [ $1 -eq 23 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 230 --lastPoint 239 -n rZbbStat.POINTS.230.239
fi
if [ $1 -eq 24 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 240 --lastPoint 249 -n rZbbStat.POINTS.240.249
fi
if [ $1 -eq 25 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 250 --lastPoint 259 -n rZbbStat.POINTS.250.259
fi
if [ $1 -eq 26 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 260 --lastPoint 269 -n rZbbStat.POINTS.260.269
fi
if [ $1 -eq 27 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 270 --lastPoint 279 -n rZbbStat.POINTS.270.279
fi
if [ $1 -eq 28 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 280 --lastPoint 289 -n rZbbStat.POINTS.280.289
fi
if [ $1 -eq 29 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 290 --lastPoint 299 -n rZbbStat.POINTS.290.299
fi
if [ $1 -eq 30 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 300 --lastPoint 309 -n rZbbStat.POINTS.300.309
fi
if [ $1 -eq 31 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 310 --lastPoint 319 -n rZbbStat.POINTS.310.319
fi
if [ $1 -eq 32 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 320 --lastPoint 329 -n rZbbStat.POINTS.320.329
fi
if [ $1 -eq 33 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 330 --lastPoint 339 -n rZbbStat.POINTS.330.339
fi
if [ $1 -eq 34 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 340 --lastPoint 349 -n rZbbStat.POINTS.340.349
fi
if [ $1 -eq 35 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 350 --lastPoint 359 -n rZbbStat.POINTS.350.359
fi
if [ $1 -eq 36 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 360 --lastPoint 369 -n rZbbStat.POINTS.360.369
fi
if [ $1 -eq 37 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 370 --lastPoint 379 -n rZbbStat.POINTS.370.379
fi
if [ $1 -eq 38 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 380 --lastPoint 389 -n rZbbStat.POINTS.380.389
fi
if [ $1 -eq 39 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 390 --lastPoint 399 -n rZbbStat.POINTS.390.399
fi
if [ $1 -eq 40 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 400 --lastPoint 409 -n rZbbStat.POINTS.400.409
fi
if [ $1 -eq 41 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 410 --lastPoint 419 -n rZbbStat.POINTS.410.419
fi
if [ $1 -eq 42 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 420 --lastPoint 429 -n rZbbStat.POINTS.420.429
fi
if [ $1 -eq 43 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 430 --lastPoint 439 -n rZbbStat.POINTS.430.439
fi
if [ $1 -eq 44 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 440 --lastPoint 449 -n rZbbStat.POINTS.440.449
fi
if [ $1 -eq 45 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 450 --lastPoint 459 -n rZbbStat.POINTS.450.459
fi
if [ $1 -eq 46 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 460 --lastPoint 469 -n rZbbStat.POINTS.460.469
fi
if [ $1 -eq 47 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 470 --lastPoint 479 -n rZbbStat.POINTS.470.479
fi
if [ $1 -eq 48 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 480 --lastPoint 489 -n rZbbStat.POINTS.480.489
fi
if [ $1 -eq 49 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 490 --lastPoint 499 -n rZbbStat.POINTS.490.499
fi
if [ $1 -eq 50 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 500 --lastPoint 509 -n rZbbStat.POINTS.500.509
fi
if [ $1 -eq 51 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 510 --lastPoint 519 -n rZbbStat.POINTS.510.519
fi
if [ $1 -eq 52 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 520 --lastPoint 529 -n rZbbStat.POINTS.520.529
fi
if [ $1 -eq 53 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 530 --lastPoint 539 -n rZbbStat.POINTS.530.539
fi
if [ $1 -eq 54 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 540 --lastPoint 549 -n rZbbStat.POINTS.540.549
fi
if [ $1 -eq 55 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 550 --lastPoint 559 -n rZbbStat.POINTS.550.559
fi
if [ $1 -eq 56 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 560 --lastPoint 569 -n rZbbStat.POINTS.560.569
fi
if [ $1 -eq 57 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 570 --lastPoint 579 -n rZbbStat.POINTS.570.579
fi
if [ $1 -eq 58 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 580 --lastPoint 589 -n rZbbStat.POINTS.580.589
fi
if [ $1 -eq 59 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 590 --lastPoint 599 -n rZbbStat.POINTS.590.599
fi
if [ $1 -eq 60 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 600 --lastPoint 609 -n rZbbStat.POINTS.600.609
fi
if [ $1 -eq 61 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 610 --lastPoint 619 -n rZbbStat.POINTS.610.619
fi
if [ $1 -eq 62 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 620 --lastPoint 629 -n rZbbStat.POINTS.620.629
fi
if [ $1 -eq 63 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 630 --lastPoint 639 -n rZbbStat.POINTS.630.639
fi
if [ $1 -eq 64 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 640 --lastPoint 649 -n rZbbStat.POINTS.640.649
fi
if [ $1 -eq 65 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 650 --lastPoint 659 -n rZbbStat.POINTS.650.659
fi
if [ $1 -eq 66 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 660 --lastPoint 669 -n rZbbStat.POINTS.660.669
fi
if [ $1 -eq 67 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 670 --lastPoint 679 -n rZbbStat.POINTS.670.679
fi
if [ $1 -eq 68 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 680 --lastPoint 689 -n rZbbStat.POINTS.680.689
fi
if [ $1 -eq 69 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 690 --lastPoint 699 -n rZbbStat.POINTS.690.699
fi
if [ $1 -eq 70 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 700 --lastPoint 709 -n rZbbStat.POINTS.700.709
fi
if [ $1 -eq 71 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 710 --lastPoint 719 -n rZbbStat.POINTS.710.719
fi
if [ $1 -eq 72 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 720 --lastPoint 729 -n rZbbStat.POINTS.720.729
fi
if [ $1 -eq 73 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 730 --lastPoint 739 -n rZbbStat.POINTS.730.739
fi
if [ $1 -eq 74 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 740 --lastPoint 749 -n rZbbStat.POINTS.740.749
fi
if [ $1 -eq 75 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 750 --lastPoint 759 -n rZbbStat.POINTS.750.759
fi
if [ $1 -eq 76 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 760 --lastPoint 769 -n rZbbStat.POINTS.760.769
fi
if [ $1 -eq 77 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 770 --lastPoint 779 -n rZbbStat.POINTS.770.779
fi
if [ $1 -eq 78 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 780 --lastPoint 789 -n rZbbStat.POINTS.780.789
fi
if [ $1 -eq 79 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 790 --lastPoint 799 -n rZbbStat.POINTS.790.799
fi
if [ $1 -eq 80 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 800 --lastPoint 809 -n rZbbStat.POINTS.800.809
fi
if [ $1 -eq 81 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 810 --lastPoint 819 -n rZbbStat.POINTS.810.819
fi
if [ $1 -eq 82 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 820 --lastPoint 829 -n rZbbStat.POINTS.820.829
fi
if [ $1 -eq 83 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 830 --lastPoint 839 -n rZbbStat.POINTS.830.839
fi
if [ $1 -eq 84 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 840 --lastPoint 849 -n rZbbStat.POINTS.840.849
fi
if [ $1 -eq 85 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 850 --lastPoint 859 -n rZbbStat.POINTS.850.859
fi
if [ $1 -eq 86 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 860 --lastPoint 869 -n rZbbStat.POINTS.860.869
fi
if [ $1 -eq 87 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 870 --lastPoint 879 -n rZbbStat.POINTS.870.879
fi
if [ $1 -eq 88 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 880 --lastPoint 889 -n rZbbStat.POINTS.880.889
fi
if [ $1 -eq 89 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 890 --lastPoint 899 -n rZbbStat.POINTS.890.899
fi
if [ $1 -eq 90 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 900 --lastPoint 909 -n rZbbStat.POINTS.900.909
fi
if [ $1 -eq 91 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 910 --lastPoint 919 -n rZbbStat.POINTS.910.919
fi
if [ $1 -eq 92 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 920 --lastPoint 929 -n rZbbStat.POINTS.920.929
fi
if [ $1 -eq 93 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 930 --lastPoint 939 -n rZbbStat.POINTS.930.939
fi
if [ $1 -eq 94 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 940 --lastPoint 949 -n rZbbStat.POINTS.940.949
fi
if [ $1 -eq 95 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 950 --lastPoint 959 -n rZbbStat.POINTS.950.959
fi
if [ $1 -eq 96 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 960 --lastPoint 969 -n rZbbStat.POINTS.960.969
fi
if [ $1 -eq 97 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 970 --lastPoint 979 -n rZbbStat.POINTS.970.979
fi
if [ $1 -eq 98 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 980 --lastPoint 989 -n rZbbStat.POINTS.980.989
fi
if [ $1 -eq 99 ]; then
  combine output/testModel/higgsCombineTest.MultiDimFit.mH125.root --algo grid --redefineSignalPOI rZbb -w w --snapshotName MultiDimFit --freezeParameters allConstrainedNuisances -M MultiDimFit --points 1000 --firstPoint 990 --lastPoint 999 -n rZbbStat.POINTS.990.999
fi