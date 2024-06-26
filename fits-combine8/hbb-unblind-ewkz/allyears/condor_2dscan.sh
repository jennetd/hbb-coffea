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
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 0 --lastPoint 9 -n 2dscan.POINTS.0.9
fi
if [ $1 -eq 1 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 10 --lastPoint 19 -n 2dscan.POINTS.10.19
fi
if [ $1 -eq 2 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 20 --lastPoint 29 -n 2dscan.POINTS.20.29
fi
if [ $1 -eq 3 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 30 --lastPoint 39 -n 2dscan.POINTS.30.39
fi
if [ $1 -eq 4 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 40 --lastPoint 49 -n 2dscan.POINTS.40.49
fi
if [ $1 -eq 5 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 50 --lastPoint 59 -n 2dscan.POINTS.50.59
fi
if [ $1 -eq 6 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 60 --lastPoint 69 -n 2dscan.POINTS.60.69
fi
if [ $1 -eq 7 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 70 --lastPoint 79 -n 2dscan.POINTS.70.79
fi
if [ $1 -eq 8 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 80 --lastPoint 89 -n 2dscan.POINTS.80.89
fi
if [ $1 -eq 9 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 90 --lastPoint 99 -n 2dscan.POINTS.90.99
fi
if [ $1 -eq 10 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 100 --lastPoint 109 -n 2dscan.POINTS.100.109
fi
if [ $1 -eq 11 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 110 --lastPoint 119 -n 2dscan.POINTS.110.119
fi
if [ $1 -eq 12 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 120 --lastPoint 129 -n 2dscan.POINTS.120.129
fi
if [ $1 -eq 13 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 130 --lastPoint 139 -n 2dscan.POINTS.130.139
fi
if [ $1 -eq 14 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 140 --lastPoint 149 -n 2dscan.POINTS.140.149
fi
if [ $1 -eq 15 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 150 --lastPoint 159 -n 2dscan.POINTS.150.159
fi
if [ $1 -eq 16 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 160 --lastPoint 169 -n 2dscan.POINTS.160.169
fi
if [ $1 -eq 17 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 170 --lastPoint 179 -n 2dscan.POINTS.170.179
fi
if [ $1 -eq 18 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 180 --lastPoint 189 -n 2dscan.POINTS.180.189
fi
if [ $1 -eq 19 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 190 --lastPoint 199 -n 2dscan.POINTS.190.199
fi
if [ $1 -eq 20 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 200 --lastPoint 209 -n 2dscan.POINTS.200.209
fi
if [ $1 -eq 21 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 210 --lastPoint 219 -n 2dscan.POINTS.210.219
fi
if [ $1 -eq 22 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 220 --lastPoint 229 -n 2dscan.POINTS.220.229
fi
if [ $1 -eq 23 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 230 --lastPoint 239 -n 2dscan.POINTS.230.239
fi
if [ $1 -eq 24 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 240 --lastPoint 249 -n 2dscan.POINTS.240.249
fi
if [ $1 -eq 25 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 250 --lastPoint 259 -n 2dscan.POINTS.250.259
fi
if [ $1 -eq 26 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 260 --lastPoint 269 -n 2dscan.POINTS.260.269
fi
if [ $1 -eq 27 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 270 --lastPoint 279 -n 2dscan.POINTS.270.279
fi
if [ $1 -eq 28 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 280 --lastPoint 289 -n 2dscan.POINTS.280.289
fi
if [ $1 -eq 29 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 290 --lastPoint 299 -n 2dscan.POINTS.290.299
fi
if [ $1 -eq 30 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 300 --lastPoint 309 -n 2dscan.POINTS.300.309
fi
if [ $1 -eq 31 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 310 --lastPoint 319 -n 2dscan.POINTS.310.319
fi
if [ $1 -eq 32 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 320 --lastPoint 329 -n 2dscan.POINTS.320.329
fi
if [ $1 -eq 33 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 330 --lastPoint 339 -n 2dscan.POINTS.330.339
fi
if [ $1 -eq 34 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 340 --lastPoint 349 -n 2dscan.POINTS.340.349
fi
if [ $1 -eq 35 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 350 --lastPoint 359 -n 2dscan.POINTS.350.359
fi
if [ $1 -eq 36 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 360 --lastPoint 369 -n 2dscan.POINTS.360.369
fi
if [ $1 -eq 37 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 370 --lastPoint 379 -n 2dscan.POINTS.370.379
fi
if [ $1 -eq 38 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 380 --lastPoint 389 -n 2dscan.POINTS.380.389
fi
if [ $1 -eq 39 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 390 --lastPoint 399 -n 2dscan.POINTS.390.399
fi
if [ $1 -eq 40 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 400 --lastPoint 409 -n 2dscan.POINTS.400.409
fi
if [ $1 -eq 41 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 410 --lastPoint 419 -n 2dscan.POINTS.410.419
fi
if [ $1 -eq 42 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 420 --lastPoint 429 -n 2dscan.POINTS.420.429
fi
if [ $1 -eq 43 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 430 --lastPoint 439 -n 2dscan.POINTS.430.439
fi
if [ $1 -eq 44 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 440 --lastPoint 449 -n 2dscan.POINTS.440.449
fi
if [ $1 -eq 45 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 450 --lastPoint 459 -n 2dscan.POINTS.450.459
fi
if [ $1 -eq 46 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 460 --lastPoint 469 -n 2dscan.POINTS.460.469
fi
if [ $1 -eq 47 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 470 --lastPoint 479 -n 2dscan.POINTS.470.479
fi
if [ $1 -eq 48 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 480 --lastPoint 489 -n 2dscan.POINTS.480.489
fi
if [ $1 -eq 49 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 490 --lastPoint 499 -n 2dscan.POINTS.490.499
fi
if [ $1 -eq 50 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 500 --lastPoint 509 -n 2dscan.POINTS.500.509
fi
if [ $1 -eq 51 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 510 --lastPoint 519 -n 2dscan.POINTS.510.519
fi
if [ $1 -eq 52 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 520 --lastPoint 529 -n 2dscan.POINTS.520.529
fi
if [ $1 -eq 53 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 530 --lastPoint 539 -n 2dscan.POINTS.530.539
fi
if [ $1 -eq 54 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 540 --lastPoint 549 -n 2dscan.POINTS.540.549
fi
if [ $1 -eq 55 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 550 --lastPoint 559 -n 2dscan.POINTS.550.559
fi
if [ $1 -eq 56 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 560 --lastPoint 569 -n 2dscan.POINTS.560.569
fi
if [ $1 -eq 57 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 570 --lastPoint 579 -n 2dscan.POINTS.570.579
fi
if [ $1 -eq 58 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 580 --lastPoint 589 -n 2dscan.POINTS.580.589
fi
if [ $1 -eq 59 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 590 --lastPoint 599 -n 2dscan.POINTS.590.599
fi
if [ $1 -eq 60 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 600 --lastPoint 609 -n 2dscan.POINTS.600.609
fi
if [ $1 -eq 61 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 610 --lastPoint 619 -n 2dscan.POINTS.610.619
fi
if [ $1 -eq 62 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 620 --lastPoint 629 -n 2dscan.POINTS.620.629
fi
if [ $1 -eq 63 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 630 --lastPoint 639 -n 2dscan.POINTS.630.639
fi
if [ $1 -eq 64 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 640 --lastPoint 649 -n 2dscan.POINTS.640.649
fi
if [ $1 -eq 65 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 650 --lastPoint 659 -n 2dscan.POINTS.650.659
fi
if [ $1 -eq 66 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 660 --lastPoint 669 -n 2dscan.POINTS.660.669
fi
if [ $1 -eq 67 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 670 --lastPoint 679 -n 2dscan.POINTS.670.679
fi
if [ $1 -eq 68 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 680 --lastPoint 689 -n 2dscan.POINTS.680.689
fi
if [ $1 -eq 69 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 690 --lastPoint 699 -n 2dscan.POINTS.690.699
fi
if [ $1 -eq 70 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 700 --lastPoint 709 -n 2dscan.POINTS.700.709
fi
if [ $1 -eq 71 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 710 --lastPoint 719 -n 2dscan.POINTS.710.719
fi
if [ $1 -eq 72 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 720 --lastPoint 729 -n 2dscan.POINTS.720.729
fi
if [ $1 -eq 73 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 730 --lastPoint 739 -n 2dscan.POINTS.730.739
fi
if [ $1 -eq 74 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 740 --lastPoint 749 -n 2dscan.POINTS.740.749
fi
if [ $1 -eq 75 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 750 --lastPoint 759 -n 2dscan.POINTS.750.759
fi
if [ $1 -eq 76 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 760 --lastPoint 769 -n 2dscan.POINTS.760.769
fi
if [ $1 -eq 77 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 770 --lastPoint 779 -n 2dscan.POINTS.770.779
fi
if [ $1 -eq 78 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 780 --lastPoint 789 -n 2dscan.POINTS.780.789
fi
if [ $1 -eq 79 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 790 --lastPoint 799 -n 2dscan.POINTS.790.799
fi
if [ $1 -eq 80 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 800 --lastPoint 809 -n 2dscan.POINTS.800.809
fi
if [ $1 -eq 81 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 810 --lastPoint 819 -n 2dscan.POINTS.810.819
fi
if [ $1 -eq 82 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 820 --lastPoint 829 -n 2dscan.POINTS.820.829
fi
if [ $1 -eq 83 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 830 --lastPoint 839 -n 2dscan.POINTS.830.839
fi
if [ $1 -eq 84 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 840 --lastPoint 849 -n 2dscan.POINTS.840.849
fi
if [ $1 -eq 85 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 850 --lastPoint 859 -n 2dscan.POINTS.850.859
fi
if [ $1 -eq 86 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 860 --lastPoint 869 -n 2dscan.POINTS.860.869
fi
if [ $1 -eq 87 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 870 --lastPoint 879 -n 2dscan.POINTS.870.879
fi
if [ $1 -eq 88 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 880 --lastPoint 889 -n 2dscan.POINTS.880.889
fi
if [ $1 -eq 89 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 890 --lastPoint 899 -n 2dscan.POINTS.890.899
fi
if [ $1 -eq 90 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 900 --lastPoint 909 -n 2dscan.POINTS.900.909
fi
if [ $1 -eq 91 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 910 --lastPoint 919 -n 2dscan.POINTS.910.919
fi
if [ $1 -eq 92 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 920 --lastPoint 929 -n 2dscan.POINTS.920.929
fi
if [ $1 -eq 93 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 930 --lastPoint 939 -n 2dscan.POINTS.930.939
fi
if [ $1 -eq 94 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 940 --lastPoint 949 -n 2dscan.POINTS.940.949
fi
if [ $1 -eq 95 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 950 --lastPoint 959 -n 2dscan.POINTS.950.959
fi
if [ $1 -eq 96 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 960 --lastPoint 969 -n 2dscan.POINTS.960.969
fi
if [ $1 -eq 97 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 970 --lastPoint 979 -n 2dscan.POINTS.970.979
fi
if [ $1 -eq 98 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 980 --lastPoint 989 -n 2dscan.POINTS.980.989
fi
if [ $1 -eq 99 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 990 --lastPoint 999 -n 2dscan.POINTS.990.999
fi
if [ $1 -eq 100 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1000 --lastPoint 1009 -n 2dscan.POINTS.1000.1009
fi
if [ $1 -eq 101 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1010 --lastPoint 1019 -n 2dscan.POINTS.1010.1019
fi
if [ $1 -eq 102 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1020 --lastPoint 1029 -n 2dscan.POINTS.1020.1029
fi
if [ $1 -eq 103 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1030 --lastPoint 1039 -n 2dscan.POINTS.1030.1039
fi
if [ $1 -eq 104 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1040 --lastPoint 1049 -n 2dscan.POINTS.1040.1049
fi
if [ $1 -eq 105 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1050 --lastPoint 1059 -n 2dscan.POINTS.1050.1059
fi
if [ $1 -eq 106 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1060 --lastPoint 1069 -n 2dscan.POINTS.1060.1069
fi
if [ $1 -eq 107 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1070 --lastPoint 1079 -n 2dscan.POINTS.1070.1079
fi
if [ $1 -eq 108 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1080 --lastPoint 1089 -n 2dscan.POINTS.1080.1089
fi
if [ $1 -eq 109 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1090 --lastPoint 1099 -n 2dscan.POINTS.1090.1099
fi
if [ $1 -eq 110 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1100 --lastPoint 1109 -n 2dscan.POINTS.1100.1109
fi
if [ $1 -eq 111 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1110 --lastPoint 1119 -n 2dscan.POINTS.1110.1119
fi
if [ $1 -eq 112 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1120 --lastPoint 1129 -n 2dscan.POINTS.1120.1129
fi
if [ $1 -eq 113 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1130 --lastPoint 1139 -n 2dscan.POINTS.1130.1139
fi
if [ $1 -eq 114 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1140 --lastPoint 1149 -n 2dscan.POINTS.1140.1149
fi
if [ $1 -eq 115 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1150 --lastPoint 1159 -n 2dscan.POINTS.1150.1159
fi
if [ $1 -eq 116 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1160 --lastPoint 1169 -n 2dscan.POINTS.1160.1169
fi
if [ $1 -eq 117 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1170 --lastPoint 1179 -n 2dscan.POINTS.1170.1179
fi
if [ $1 -eq 118 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1180 --lastPoint 1189 -n 2dscan.POINTS.1180.1189
fi
if [ $1 -eq 119 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1190 --lastPoint 1199 -n 2dscan.POINTS.1190.1199
fi
if [ $1 -eq 120 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1200 --lastPoint 1209 -n 2dscan.POINTS.1200.1209
fi
if [ $1 -eq 121 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1210 --lastPoint 1219 -n 2dscan.POINTS.1210.1219
fi
if [ $1 -eq 122 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1220 --lastPoint 1229 -n 2dscan.POINTS.1220.1229
fi
if [ $1 -eq 123 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1230 --lastPoint 1239 -n 2dscan.POINTS.1230.1239
fi
if [ $1 -eq 124 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1240 --lastPoint 1249 -n 2dscan.POINTS.1240.1249
fi
if [ $1 -eq 125 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1250 --lastPoint 1259 -n 2dscan.POINTS.1250.1259
fi
if [ $1 -eq 126 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1260 --lastPoint 1269 -n 2dscan.POINTS.1260.1269
fi
if [ $1 -eq 127 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1270 --lastPoint 1279 -n 2dscan.POINTS.1270.1279
fi
if [ $1 -eq 128 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1280 --lastPoint 1289 -n 2dscan.POINTS.1280.1289
fi
if [ $1 -eq 129 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1290 --lastPoint 1299 -n 2dscan.POINTS.1290.1299
fi
if [ $1 -eq 130 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1300 --lastPoint 1309 -n 2dscan.POINTS.1300.1309
fi
if [ $1 -eq 131 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1310 --lastPoint 1319 -n 2dscan.POINTS.1310.1319
fi
if [ $1 -eq 132 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1320 --lastPoint 1329 -n 2dscan.POINTS.1320.1329
fi
if [ $1 -eq 133 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1330 --lastPoint 1339 -n 2dscan.POINTS.1330.1339
fi
if [ $1 -eq 134 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1340 --lastPoint 1349 -n 2dscan.POINTS.1340.1349
fi
if [ $1 -eq 135 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1350 --lastPoint 1359 -n 2dscan.POINTS.1350.1359
fi
if [ $1 -eq 136 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1360 --lastPoint 1369 -n 2dscan.POINTS.1360.1369
fi
if [ $1 -eq 137 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1370 --lastPoint 1379 -n 2dscan.POINTS.1370.1379
fi
if [ $1 -eq 138 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1380 --lastPoint 1389 -n 2dscan.POINTS.1380.1389
fi
if [ $1 -eq 139 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1390 --lastPoint 1399 -n 2dscan.POINTS.1390.1399
fi
if [ $1 -eq 140 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1400 --lastPoint 1409 -n 2dscan.POINTS.1400.1409
fi
if [ $1 -eq 141 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1410 --lastPoint 1419 -n 2dscan.POINTS.1410.1419
fi
if [ $1 -eq 142 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1420 --lastPoint 1429 -n 2dscan.POINTS.1420.1429
fi
if [ $1 -eq 143 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1430 --lastPoint 1439 -n 2dscan.POINTS.1430.1439
fi
if [ $1 -eq 144 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1440 --lastPoint 1449 -n 2dscan.POINTS.1440.1449
fi
if [ $1 -eq 145 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1450 --lastPoint 1459 -n 2dscan.POINTS.1450.1459
fi
if [ $1 -eq 146 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1460 --lastPoint 1469 -n 2dscan.POINTS.1460.1469
fi
if [ $1 -eq 147 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1470 --lastPoint 1479 -n 2dscan.POINTS.1470.1479
fi
if [ $1 -eq 148 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1480 --lastPoint 1489 -n 2dscan.POINTS.1480.1489
fi
if [ $1 -eq 149 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1490 --lastPoint 1499 -n 2dscan.POINTS.1490.1499
fi
if [ $1 -eq 150 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1500 --lastPoint 1509 -n 2dscan.POINTS.1500.1509
fi
if [ $1 -eq 151 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1510 --lastPoint 1519 -n 2dscan.POINTS.1510.1519
fi
if [ $1 -eq 152 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1520 --lastPoint 1529 -n 2dscan.POINTS.1520.1529
fi
if [ $1 -eq 153 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1530 --lastPoint 1539 -n 2dscan.POINTS.1530.1539
fi
if [ $1 -eq 154 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1540 --lastPoint 1549 -n 2dscan.POINTS.1540.1549
fi
if [ $1 -eq 155 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1550 --lastPoint 1559 -n 2dscan.POINTS.1550.1559
fi
if [ $1 -eq 156 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1560 --lastPoint 1569 -n 2dscan.POINTS.1560.1569
fi
if [ $1 -eq 157 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1570 --lastPoint 1579 -n 2dscan.POINTS.1570.1579
fi
if [ $1 -eq 158 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1580 --lastPoint 1589 -n 2dscan.POINTS.1580.1589
fi
if [ $1 -eq 159 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1590 --lastPoint 1599 -n 2dscan.POINTS.1590.1599
fi
if [ $1 -eq 160 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1600 --lastPoint 1609 -n 2dscan.POINTS.1600.1609
fi
if [ $1 -eq 161 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1610 --lastPoint 1619 -n 2dscan.POINTS.1610.1619
fi
if [ $1 -eq 162 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1620 --lastPoint 1629 -n 2dscan.POINTS.1620.1629
fi
if [ $1 -eq 163 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1630 --lastPoint 1639 -n 2dscan.POINTS.1630.1639
fi
if [ $1 -eq 164 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1640 --lastPoint 1649 -n 2dscan.POINTS.1640.1649
fi
if [ $1 -eq 165 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1650 --lastPoint 1659 -n 2dscan.POINTS.1650.1659
fi
if [ $1 -eq 166 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1660 --lastPoint 1669 -n 2dscan.POINTS.1660.1669
fi
if [ $1 -eq 167 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1670 --lastPoint 1679 -n 2dscan.POINTS.1670.1679
fi
if [ $1 -eq 168 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1680 --lastPoint 1689 -n 2dscan.POINTS.1680.1689
fi
if [ $1 -eq 169 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1690 --lastPoint 1699 -n 2dscan.POINTS.1690.1699
fi
if [ $1 -eq 170 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1700 --lastPoint 1709 -n 2dscan.POINTS.1700.1709
fi
if [ $1 -eq 171 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1710 --lastPoint 1719 -n 2dscan.POINTS.1710.1719
fi
if [ $1 -eq 172 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1720 --lastPoint 1729 -n 2dscan.POINTS.1720.1729
fi
if [ $1 -eq 173 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1730 --lastPoint 1739 -n 2dscan.POINTS.1730.1739
fi
if [ $1 -eq 174 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1740 --lastPoint 1749 -n 2dscan.POINTS.1740.1749
fi
if [ $1 -eq 175 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1750 --lastPoint 1759 -n 2dscan.POINTS.1750.1759
fi
if [ $1 -eq 176 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1760 --lastPoint 1769 -n 2dscan.POINTS.1760.1769
fi
if [ $1 -eq 177 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1770 --lastPoint 1779 -n 2dscan.POINTS.1770.1779
fi
if [ $1 -eq 178 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1780 --lastPoint 1789 -n 2dscan.POINTS.1780.1789
fi
if [ $1 -eq 179 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1790 --lastPoint 1799 -n 2dscan.POINTS.1790.1799
fi
if [ $1 -eq 180 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1800 --lastPoint 1809 -n 2dscan.POINTS.1800.1809
fi
if [ $1 -eq 181 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1810 --lastPoint 1819 -n 2dscan.POINTS.1810.1819
fi
if [ $1 -eq 182 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1820 --lastPoint 1829 -n 2dscan.POINTS.1820.1829
fi
if [ $1 -eq 183 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1830 --lastPoint 1839 -n 2dscan.POINTS.1830.1839
fi
if [ $1 -eq 184 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1840 --lastPoint 1849 -n 2dscan.POINTS.1840.1849
fi
if [ $1 -eq 185 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1850 --lastPoint 1859 -n 2dscan.POINTS.1850.1859
fi
if [ $1 -eq 186 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1860 --lastPoint 1869 -n 2dscan.POINTS.1860.1869
fi
if [ $1 -eq 187 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1870 --lastPoint 1879 -n 2dscan.POINTS.1870.1879
fi
if [ $1 -eq 188 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1880 --lastPoint 1889 -n 2dscan.POINTS.1880.1889
fi
if [ $1 -eq 189 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1890 --lastPoint 1899 -n 2dscan.POINTS.1890.1899
fi
if [ $1 -eq 190 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1900 --lastPoint 1909 -n 2dscan.POINTS.1900.1909
fi
if [ $1 -eq 191 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1910 --lastPoint 1919 -n 2dscan.POINTS.1910.1919
fi
if [ $1 -eq 192 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1920 --lastPoint 1929 -n 2dscan.POINTS.1920.1929
fi
if [ $1 -eq 193 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1930 --lastPoint 1939 -n 2dscan.POINTS.1930.1939
fi
if [ $1 -eq 194 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1940 --lastPoint 1949 -n 2dscan.POINTS.1940.1949
fi
if [ $1 -eq 195 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1950 --lastPoint 1959 -n 2dscan.POINTS.1950.1959
fi
if [ $1 -eq 196 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1960 --lastPoint 1969 -n 2dscan.POINTS.1960.1969
fi
if [ $1 -eq 197 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1970 --lastPoint 1979 -n 2dscan.POINTS.1970.1979
fi
if [ $1 -eq 198 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1980 --lastPoint 1989 -n 2dscan.POINTS.1980.1989
fi
if [ $1 -eq 199 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 1990 --lastPoint 1999 -n 2dscan.POINTS.1990.1999
fi
if [ $1 -eq 200 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2000 --lastPoint 2009 -n 2dscan.POINTS.2000.2009
fi
if [ $1 -eq 201 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2010 --lastPoint 2019 -n 2dscan.POINTS.2010.2019
fi
if [ $1 -eq 202 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2020 --lastPoint 2029 -n 2dscan.POINTS.2020.2029
fi
if [ $1 -eq 203 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2030 --lastPoint 2039 -n 2dscan.POINTS.2030.2039
fi
if [ $1 -eq 204 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2040 --lastPoint 2049 -n 2dscan.POINTS.2040.2049
fi
if [ $1 -eq 205 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2050 --lastPoint 2059 -n 2dscan.POINTS.2050.2059
fi
if [ $1 -eq 206 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2060 --lastPoint 2069 -n 2dscan.POINTS.2060.2069
fi
if [ $1 -eq 207 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2070 --lastPoint 2079 -n 2dscan.POINTS.2070.2079
fi
if [ $1 -eq 208 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2080 --lastPoint 2089 -n 2dscan.POINTS.2080.2089
fi
if [ $1 -eq 209 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2090 --lastPoint 2099 -n 2dscan.POINTS.2090.2099
fi
if [ $1 -eq 210 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2100 --lastPoint 2109 -n 2dscan.POINTS.2100.2109
fi
if [ $1 -eq 211 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2110 --lastPoint 2119 -n 2dscan.POINTS.2110.2119
fi
if [ $1 -eq 212 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2120 --lastPoint 2129 -n 2dscan.POINTS.2120.2129
fi
if [ $1 -eq 213 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2130 --lastPoint 2139 -n 2dscan.POINTS.2130.2139
fi
if [ $1 -eq 214 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2140 --lastPoint 2149 -n 2dscan.POINTS.2140.2149
fi
if [ $1 -eq 215 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2150 --lastPoint 2159 -n 2dscan.POINTS.2150.2159
fi
if [ $1 -eq 216 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2160 --lastPoint 2169 -n 2dscan.POINTS.2160.2169
fi
if [ $1 -eq 217 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2170 --lastPoint 2179 -n 2dscan.POINTS.2170.2179
fi
if [ $1 -eq 218 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2180 --lastPoint 2189 -n 2dscan.POINTS.2180.2189
fi
if [ $1 -eq 219 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2190 --lastPoint 2199 -n 2dscan.POINTS.2190.2199
fi
if [ $1 -eq 220 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2200 --lastPoint 2209 -n 2dscan.POINTS.2200.2209
fi
if [ $1 -eq 221 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2210 --lastPoint 2219 -n 2dscan.POINTS.2210.2219
fi
if [ $1 -eq 222 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2220 --lastPoint 2229 -n 2dscan.POINTS.2220.2229
fi
if [ $1 -eq 223 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2230 --lastPoint 2239 -n 2dscan.POINTS.2230.2239
fi
if [ $1 -eq 224 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2240 --lastPoint 2249 -n 2dscan.POINTS.2240.2249
fi
if [ $1 -eq 225 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2250 --lastPoint 2259 -n 2dscan.POINTS.2250.2259
fi
if [ $1 -eq 226 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2260 --lastPoint 2269 -n 2dscan.POINTS.2260.2269
fi
if [ $1 -eq 227 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2270 --lastPoint 2279 -n 2dscan.POINTS.2270.2279
fi
if [ $1 -eq 228 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2280 --lastPoint 2289 -n 2dscan.POINTS.2280.2289
fi
if [ $1 -eq 229 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2290 --lastPoint 2299 -n 2dscan.POINTS.2290.2299
fi
if [ $1 -eq 230 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2300 --lastPoint 2309 -n 2dscan.POINTS.2300.2309
fi
if [ $1 -eq 231 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2310 --lastPoint 2319 -n 2dscan.POINTS.2310.2319
fi
if [ $1 -eq 232 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2320 --lastPoint 2329 -n 2dscan.POINTS.2320.2329
fi
if [ $1 -eq 233 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2330 --lastPoint 2339 -n 2dscan.POINTS.2330.2339
fi
if [ $1 -eq 234 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2340 --lastPoint 2349 -n 2dscan.POINTS.2340.2349
fi
if [ $1 -eq 235 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2350 --lastPoint 2359 -n 2dscan.POINTS.2350.2359
fi
if [ $1 -eq 236 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2360 --lastPoint 2369 -n 2dscan.POINTS.2360.2369
fi
if [ $1 -eq 237 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2370 --lastPoint 2379 -n 2dscan.POINTS.2370.2379
fi
if [ $1 -eq 238 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2380 --lastPoint 2389 -n 2dscan.POINTS.2380.2389
fi
if [ $1 -eq 239 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2390 --lastPoint 2399 -n 2dscan.POINTS.2390.2399
fi
if [ $1 -eq 240 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2400 --lastPoint 2409 -n 2dscan.POINTS.2400.2409
fi
if [ $1 -eq 241 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2410 --lastPoint 2419 -n 2dscan.POINTS.2410.2419
fi
if [ $1 -eq 242 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2420 --lastPoint 2429 -n 2dscan.POINTS.2420.2429
fi
if [ $1 -eq 243 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2430 --lastPoint 2439 -n 2dscan.POINTS.2430.2439
fi
if [ $1 -eq 244 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2440 --lastPoint 2449 -n 2dscan.POINTS.2440.2449
fi
if [ $1 -eq 245 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2450 --lastPoint 2459 -n 2dscan.POINTS.2450.2459
fi
if [ $1 -eq 246 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2460 --lastPoint 2469 -n 2dscan.POINTS.2460.2469
fi
if [ $1 -eq 247 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2470 --lastPoint 2479 -n 2dscan.POINTS.2470.2479
fi
if [ $1 -eq 248 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2480 --lastPoint 2489 -n 2dscan.POINTS.2480.2489
fi
if [ $1 -eq 249 ]; then
  combine output/testModel/model_combined.root --algo grid -M MultiDimFit --points 2500 --firstPoint 2490 --lastPoint 2499 -n 2dscan.POINTS.2490.2499
fi