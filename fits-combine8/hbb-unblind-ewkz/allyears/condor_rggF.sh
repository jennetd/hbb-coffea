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
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 0 --lastPoint 9 -n rggF.POINTS.0.9
fi
if [ $1 -eq 1 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 10 --lastPoint 19 -n rggF.POINTS.10.19
fi
if [ $1 -eq 2 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 20 --lastPoint 29 -n rggF.POINTS.20.29
fi
if [ $1 -eq 3 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 30 --lastPoint 39 -n rggF.POINTS.30.39
fi
if [ $1 -eq 4 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 40 --lastPoint 49 -n rggF.POINTS.40.49
fi
if [ $1 -eq 5 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 50 --lastPoint 59 -n rggF.POINTS.50.59
fi
if [ $1 -eq 6 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 60 --lastPoint 69 -n rggF.POINTS.60.69
fi
if [ $1 -eq 7 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 70 --lastPoint 79 -n rggF.POINTS.70.79
fi
if [ $1 -eq 8 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 80 --lastPoint 89 -n rggF.POINTS.80.89
fi
if [ $1 -eq 9 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 90 --lastPoint 99 -n rggF.POINTS.90.99
fi
if [ $1 -eq 10 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 100 --lastPoint 109 -n rggF.POINTS.100.109
fi
if [ $1 -eq 11 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 110 --lastPoint 119 -n rggF.POINTS.110.119
fi
if [ $1 -eq 12 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 120 --lastPoint 129 -n rggF.POINTS.120.129
fi
if [ $1 -eq 13 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 130 --lastPoint 139 -n rggF.POINTS.130.139
fi
if [ $1 -eq 14 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 140 --lastPoint 149 -n rggF.POINTS.140.149
fi
if [ $1 -eq 15 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 150 --lastPoint 159 -n rggF.POINTS.150.159
fi
if [ $1 -eq 16 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 160 --lastPoint 169 -n rggF.POINTS.160.169
fi
if [ $1 -eq 17 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 170 --lastPoint 179 -n rggF.POINTS.170.179
fi
if [ $1 -eq 18 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 180 --lastPoint 189 -n rggF.POINTS.180.189
fi
if [ $1 -eq 19 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 190 --lastPoint 199 -n rggF.POINTS.190.199
fi
if [ $1 -eq 20 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 200 --lastPoint 209 -n rggF.POINTS.200.209
fi
if [ $1 -eq 21 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 210 --lastPoint 219 -n rggF.POINTS.210.219
fi
if [ $1 -eq 22 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 220 --lastPoint 229 -n rggF.POINTS.220.229
fi
if [ $1 -eq 23 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 230 --lastPoint 239 -n rggF.POINTS.230.239
fi
if [ $1 -eq 24 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 240 --lastPoint 249 -n rggF.POINTS.240.249
fi
if [ $1 -eq 25 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 250 --lastPoint 259 -n rggF.POINTS.250.259
fi
if [ $1 -eq 26 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 260 --lastPoint 269 -n rggF.POINTS.260.269
fi
if [ $1 -eq 27 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 270 --lastPoint 279 -n rggF.POINTS.270.279
fi
if [ $1 -eq 28 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 280 --lastPoint 289 -n rggF.POINTS.280.289
fi
if [ $1 -eq 29 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 290 --lastPoint 299 -n rggF.POINTS.290.299
fi
if [ $1 -eq 30 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 300 --lastPoint 309 -n rggF.POINTS.300.309
fi
if [ $1 -eq 31 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 310 --lastPoint 319 -n rggF.POINTS.310.319
fi
if [ $1 -eq 32 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 320 --lastPoint 329 -n rggF.POINTS.320.329
fi
if [ $1 -eq 33 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 330 --lastPoint 339 -n rggF.POINTS.330.339
fi
if [ $1 -eq 34 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 340 --lastPoint 349 -n rggF.POINTS.340.349
fi
if [ $1 -eq 35 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 350 --lastPoint 359 -n rggF.POINTS.350.359
fi
if [ $1 -eq 36 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 360 --lastPoint 369 -n rggF.POINTS.360.369
fi
if [ $1 -eq 37 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 370 --lastPoint 379 -n rggF.POINTS.370.379
fi
if [ $1 -eq 38 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 380 --lastPoint 389 -n rggF.POINTS.380.389
fi
if [ $1 -eq 39 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 390 --lastPoint 399 -n rggF.POINTS.390.399
fi
if [ $1 -eq 40 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 400 --lastPoint 409 -n rggF.POINTS.400.409
fi
if [ $1 -eq 41 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 410 --lastPoint 419 -n rggF.POINTS.410.419
fi
if [ $1 -eq 42 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 420 --lastPoint 429 -n rggF.POINTS.420.429
fi
if [ $1 -eq 43 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 430 --lastPoint 439 -n rggF.POINTS.430.439
fi
if [ $1 -eq 44 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 440 --lastPoint 449 -n rggF.POINTS.440.449
fi
if [ $1 -eq 45 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 450 --lastPoint 459 -n rggF.POINTS.450.459
fi
if [ $1 -eq 46 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 460 --lastPoint 469 -n rggF.POINTS.460.469
fi
if [ $1 -eq 47 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 470 --lastPoint 479 -n rggF.POINTS.470.479
fi
if [ $1 -eq 48 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 480 --lastPoint 489 -n rggF.POINTS.480.489
fi
if [ $1 -eq 49 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 490 --lastPoint 499 -n rggF.POINTS.490.499
fi
if [ $1 -eq 50 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 500 --lastPoint 509 -n rggF.POINTS.500.509
fi
if [ $1 -eq 51 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 510 --lastPoint 519 -n rggF.POINTS.510.519
fi
if [ $1 -eq 52 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 520 --lastPoint 529 -n rggF.POINTS.520.529
fi
if [ $1 -eq 53 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 530 --lastPoint 539 -n rggF.POINTS.530.539
fi
if [ $1 -eq 54 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 540 --lastPoint 549 -n rggF.POINTS.540.549
fi
if [ $1 -eq 55 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 550 --lastPoint 559 -n rggF.POINTS.550.559
fi
if [ $1 -eq 56 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 560 --lastPoint 569 -n rggF.POINTS.560.569
fi
if [ $1 -eq 57 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 570 --lastPoint 579 -n rggF.POINTS.570.579
fi
if [ $1 -eq 58 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 580 --lastPoint 589 -n rggF.POINTS.580.589
fi
if [ $1 -eq 59 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 590 --lastPoint 599 -n rggF.POINTS.590.599
fi
if [ $1 -eq 60 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 600 --lastPoint 609 -n rggF.POINTS.600.609
fi
if [ $1 -eq 61 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 610 --lastPoint 619 -n rggF.POINTS.610.619
fi
if [ $1 -eq 62 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 620 --lastPoint 629 -n rggF.POINTS.620.629
fi
if [ $1 -eq 63 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 630 --lastPoint 639 -n rggF.POINTS.630.639
fi
if [ $1 -eq 64 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 640 --lastPoint 649 -n rggF.POINTS.640.649
fi
if [ $1 -eq 65 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 650 --lastPoint 659 -n rggF.POINTS.650.659
fi
if [ $1 -eq 66 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 660 --lastPoint 669 -n rggF.POINTS.660.669
fi
if [ $1 -eq 67 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 670 --lastPoint 679 -n rggF.POINTS.670.679
fi
if [ $1 -eq 68 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 680 --lastPoint 689 -n rggF.POINTS.680.689
fi
if [ $1 -eq 69 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 690 --lastPoint 699 -n rggF.POINTS.690.699
fi
if [ $1 -eq 70 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 700 --lastPoint 709 -n rggF.POINTS.700.709
fi
if [ $1 -eq 71 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 710 --lastPoint 719 -n rggF.POINTS.710.719
fi
if [ $1 -eq 72 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 720 --lastPoint 729 -n rggF.POINTS.720.729
fi
if [ $1 -eq 73 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 730 --lastPoint 739 -n rggF.POINTS.730.739
fi
if [ $1 -eq 74 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 740 --lastPoint 749 -n rggF.POINTS.740.749
fi
if [ $1 -eq 75 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 750 --lastPoint 759 -n rggF.POINTS.750.759
fi
if [ $1 -eq 76 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 760 --lastPoint 769 -n rggF.POINTS.760.769
fi
if [ $1 -eq 77 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 770 --lastPoint 779 -n rggF.POINTS.770.779
fi
if [ $1 -eq 78 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 780 --lastPoint 789 -n rggF.POINTS.780.789
fi
if [ $1 -eq 79 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 790 --lastPoint 799 -n rggF.POINTS.790.799
fi
if [ $1 -eq 80 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 800 --lastPoint 809 -n rggF.POINTS.800.809
fi
if [ $1 -eq 81 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 810 --lastPoint 819 -n rggF.POINTS.810.819
fi
if [ $1 -eq 82 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 820 --lastPoint 829 -n rggF.POINTS.820.829
fi
if [ $1 -eq 83 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 830 --lastPoint 839 -n rggF.POINTS.830.839
fi
if [ $1 -eq 84 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 840 --lastPoint 849 -n rggF.POINTS.840.849
fi
if [ $1 -eq 85 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 850 --lastPoint 859 -n rggF.POINTS.850.859
fi
if [ $1 -eq 86 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 860 --lastPoint 869 -n rggF.POINTS.860.869
fi
if [ $1 -eq 87 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 870 --lastPoint 879 -n rggF.POINTS.870.879
fi
if [ $1 -eq 88 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 880 --lastPoint 889 -n rggF.POINTS.880.889
fi
if [ $1 -eq 89 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 890 --lastPoint 899 -n rggF.POINTS.890.899
fi
if [ $1 -eq 90 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 900 --lastPoint 909 -n rggF.POINTS.900.909
fi
if [ $1 -eq 91 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 910 --lastPoint 919 -n rggF.POINTS.910.919
fi
if [ $1 -eq 92 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 920 --lastPoint 929 -n rggF.POINTS.920.929
fi
if [ $1 -eq 93 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 930 --lastPoint 939 -n rggF.POINTS.930.939
fi
if [ $1 -eq 94 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 940 --lastPoint 949 -n rggF.POINTS.940.949
fi
if [ $1 -eq 95 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 950 --lastPoint 959 -n rggF.POINTS.950.959
fi
if [ $1 -eq 96 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 960 --lastPoint 969 -n rggF.POINTS.960.969
fi
if [ $1 -eq 97 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 970 --lastPoint 979 -n rggF.POINTS.970.979
fi
if [ $1 -eq 98 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 980 --lastPoint 989 -n rggF.POINTS.980.989
fi
if [ $1 -eq 99 ]; then
  combine output/testModel/model_combined.root --algo grid --redefineSignalPOI rggF -M MultiDimFit --points 1000 --firstPoint 990 --lastPoint 999 -n rggF.POINTS.990.999
fi