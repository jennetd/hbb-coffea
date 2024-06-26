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
  combine -M MultiDimFit -n _paramFit_Test_CMS_L1Prefiring_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_L1Prefiring_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_L1Prefiring_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_L1Prefiring_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 2 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_L1Prefiring_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_L1Prefiring_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 3 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_eff_bb_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_eff_bb_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 4 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_eff_bb_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_eff_bb_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 5 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_eff_bb_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_eff_bb_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 6 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_eff_bb_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_eff_bb_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 7 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_PU_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_PU_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 8 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_PU_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_PU_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 9 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_PU_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_PU_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 10 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_PU_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_PU_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 11 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_W_d2kappa_EW --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_W_d2kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 12 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_W_d3kappa_EW --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_W_d3kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 13 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_Z_d2kappa_EW --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_Z_d2kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 14 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_Z_d3kappa_EW --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_Z_d3kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 15 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFbc_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 16 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFbc_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 17 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFbc_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 18 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFbc_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 19 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_correlated --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFbc_correlated --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 20 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFlight_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 21 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFlight_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 22 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFlight_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 23 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFlight_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 24 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_correlated --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_btagSFlight_correlated --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 25 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d1K_NLO --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_d1K_NLO --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 26 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d1kappa_EW --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_d1kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 27 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d2K_NLO --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_d2K_NLO --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 28 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d3K_NLO --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_d3K_NLO --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 29 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_1_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_1_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 30 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_1_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_1_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 31 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_1_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_1_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 32 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_1_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_1_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 33 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_2_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_2_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 34 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_2_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_2_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 35 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_2_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_2_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 36 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_2_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_2_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 37 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_3_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_3_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 38 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_3_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_3_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 39 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_3_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_3_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 40 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_3_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_ddb_3_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 41 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_e_veto_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_e_veto_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 42 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_e_veto_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_e_veto_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 43 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_e_veto_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_e_veto_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 44 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_e_veto_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_e_veto_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 45 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_jet_trigger_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_jet_trigger_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 46 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_jet_trigger_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_jet_trigger_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 47 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_jet_trigger_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_jet_trigger_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 48 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_jet_trigger_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_jet_trigger_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 49 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_trigger_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_trigger_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 50 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_trigger_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_trigger_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 51 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_trigger_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_trigger_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 52 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_trigger_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_trigger_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 53 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_veto_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_veto_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 54 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_veto_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_veto_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 55 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_veto_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_veto_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 56 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_veto_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_mu_veto_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 57 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 58 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 59 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 60 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 61 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_1_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_1_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 62 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_1_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_1_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 63 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_1_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_1_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 64 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_1_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_1_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 65 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_2_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_2_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 66 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_2_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_2_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 67 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_2_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_2_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 68 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_2_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_2_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 69 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_3_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_3_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 70 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_3_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_3_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 71 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_3_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_3_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 72 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_3_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_3_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 73 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_4_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_4_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 74 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_4_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_4_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 75 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_4_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_4_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 76 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_4_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_4_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 77 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_5_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_5_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 78 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_5_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_5_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 79 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_5_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_5_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 80 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_5_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_5_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 81 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_6_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_6_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 82 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_6_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_6_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 83 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_6_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_6_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 84 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_6_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_scale_pt_6_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 85 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_smear_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_smear_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 86 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_smear_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_smear_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 87 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_smear_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_smear_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 88 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_smear_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_smear_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 89 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_tau_veto_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_tau_veto_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 90 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_tau_veto_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_tau_veto_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 91 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_tau_veto_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_tau_veto_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 92 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_tau_veto_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_tau_veto_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 93 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_vbfmucr_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_vbfmucr_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 94 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_vbfmucr_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_vbfmucr_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 95 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_vbfmucr_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_vbfmucr_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 96 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_vbfmucr_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_vbfmucr_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 97 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_1_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_1_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 98 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_1_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_1_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 99 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_1_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_1_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 100 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_1_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_1_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 101 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 102 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 103 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 104 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 105 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 106 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 107 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 108 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_2_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 109 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_3_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_3_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 110 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_3_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_3_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 111 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_3_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_3_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 112 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_3_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_hbb_veff_3_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 113 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_lumi_13TeV_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 114 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_lumi_13TeV_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 115 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_lumi_13TeV_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 116 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_correlated --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_lumi_13TeV_correlated --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 117 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_correlated_20172018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_lumi_13TeV_correlated_20172018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 118 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_id_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_id_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 119 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_id_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_id_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 120 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_id_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_id_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 121 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_id_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_id_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 122 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_iso_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_iso_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 123 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_iso_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_iso_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 124 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_iso_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_iso_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 125 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_iso_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_mu_iso_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 126 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_res_j_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_res_j_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 127 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_res_j_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_res_j_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 128 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_res_j_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_res_j_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 129 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_res_j_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_res_j_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 130 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_scale_j_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_scale_j_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 131 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_scale_j_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_scale_j_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 132 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_scale_j_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_scale_j_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 133 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_scale_j_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_scale_j_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 134 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_ues_j_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_ues_j_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 135 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_ues_j_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_ues_j_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 136 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_ues_j_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_ues_j_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 137 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_ues_j_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P CMS_ues_j_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 138 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_EWKZ --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_EWKZ --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 139 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 140 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_VH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 141 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_acceptance_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_acceptance_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 142 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_acceptance_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_acceptance_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 143 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 144 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_ttH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P QCDscale_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 145 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_FSR_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 146 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_VH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_FSR_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 147 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_acceptance_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_FSR_acceptance_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 148 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_acceptance_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_FSR_acceptance_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 149 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_FSR_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 150 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_ttH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_FSR_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 151 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_ISR_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 152 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_VH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_ISR_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 153 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_acceptance_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_ISR_acceptance_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 154 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_acceptance_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_ISR_acceptance_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 155 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_ISR_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 156 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_ttH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P UEPS_ISR_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 157 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016APV_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016APV_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 158 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 159 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 160 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 161 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 162 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 163 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 164 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 165 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 166 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 167 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 168 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 169 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2017_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 170 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 171 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 172 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 173 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 174 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 175 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2018_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2018_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 176 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 177 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 178 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 179 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 180 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRfail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 181 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016APV_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016APV_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 182 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 183 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 184 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 185 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 186 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 187 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 188 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2017_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 189 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 190 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 191 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 192 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 193 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2018_QCD_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2018_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 194 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 195 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 196 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 197 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P muonCRpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 198 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P pdf_Higgs_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 199 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_VH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P pdf_Higgs_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 200 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P pdf_Higgs_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 201 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_ttH --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P pdf_Higgs_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 202 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_acceptance_VBF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P pdf_acceptance_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 203 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_acceptance_ggF --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P pdf_acceptance_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 204 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 205 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 206 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 207 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 208 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 209 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 210 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 211 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 212 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 213 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 214 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 215 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 216 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 217 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 218 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 219 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 220 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 221 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 222 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 223 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 224 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 225 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 226 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 227 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 228 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 229 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 230 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 231 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 232 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 233 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 234 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 235 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 236 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 237 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 238 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 239 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 240 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 241 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 242 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 243 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 244 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 245 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 246 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 247 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 248 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 249 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 250 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 251 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 252 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 253 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 254 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 255 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 256 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 257 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 258 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 259 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 260 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 261 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 262 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 263 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 264 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 265 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 266 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 267 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 268 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 269 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 270 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 271 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 272 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 273 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 274 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 275 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 276 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 277 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggffail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 278 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 279 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 280 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 281 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 282 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 283 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 284 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 285 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 286 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 287 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 288 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 289 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 290 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 291 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 292 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 293 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 294 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 295 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 296 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 297 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 298 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 299 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 300 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 301 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 302 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 303 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 304 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 305 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 306 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 307 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 308 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 309 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 310 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 311 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 312 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 313 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 314 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 315 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 316 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 317 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 318 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 319 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 320 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 321 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 322 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 323 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 324 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 325 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 326 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 327 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 328 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 329 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 330 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 331 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 332 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 333 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 334 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 335 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 336 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 337 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 338 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 339 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 340 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 341 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 342 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 343 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 344 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 345 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0ggfpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 346 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 347 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 348 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 349 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 350 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 351 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 352 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 353 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 354 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 355 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 356 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 357 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 358 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 359 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 360 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 361 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 362 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 363 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 364 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 365 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 366 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 367 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 368 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 369 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 370 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 371 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 372 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 373 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 374 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 375 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 376 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 377 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 378 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 379 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 380 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 381 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 382 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 383 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 384 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 385 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 386 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 387 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 388 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 389 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 390 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 391 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 392 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 393 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 394 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 395 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 396 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 397 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 398 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 399 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 400 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 401 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 402 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 403 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 404 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 405 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 406 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 407 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 408 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 409 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 410 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 411 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 412 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 413 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhifail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 414 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 415 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 416 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 417 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 418 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 419 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 420 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 421 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 422 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 423 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 424 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 425 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 426 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 427 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 428 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 429 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 430 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 431 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 432 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 433 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 434 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 435 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 436 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 437 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 438 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 439 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 440 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 441 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 442 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 443 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 444 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 445 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 446 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 447 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 448 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 449 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 450 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 451 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 452 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 453 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 454 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 455 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 456 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 457 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 458 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 459 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 460 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 461 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 462 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 463 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 464 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 465 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 466 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 467 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 468 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 469 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 470 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 471 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 472 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 473 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 474 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 475 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 476 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 477 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbfhipass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 478 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 479 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 480 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 481 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 482 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 483 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 484 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 485 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 486 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 487 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 488 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 489 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 490 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 491 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 492 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 493 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 494 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 495 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 496 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 497 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 498 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 499 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 500 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 501 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 502 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 503 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 504 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 505 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 506 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 507 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 508 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 509 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 510 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 511 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 512 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 513 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 514 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 515 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 516 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 517 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 518 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 519 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 520 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 521 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 522 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 523 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 524 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 525 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 526 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 527 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 528 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 529 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 530 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 531 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 532 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 533 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 534 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 535 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 536 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 537 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 538 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 539 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 540 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 541 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 542 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 543 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 544 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 545 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 546 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 547 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 548 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 549 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflofail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 550 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 551 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 552 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 553 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 554 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 555 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 556 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 557 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 558 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 559 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 560 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 561 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 562 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 563 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 564 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 565 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 566 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 567 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 568 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 569 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 570 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 571 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 572 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 573 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 574 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 575 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 576 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 577 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 578 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 579 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 580 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 581 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 582 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 583 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 584 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 585 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 586 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 587 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 588 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 589 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 590 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 591 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 592 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 593 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 594 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 595 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 596 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 597 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 598 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 599 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 600 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 601 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 602 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 603 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 604 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 605 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 606 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 607 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 608 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 609 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 610 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 611 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 612 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 613 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 614 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 615 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 616 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 617 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 618 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 619 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin0vbflopass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 620 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 621 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 622 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 623 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 624 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 625 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 626 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 627 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 628 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 629 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 630 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 631 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 632 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 633 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 634 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 635 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 636 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 637 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 638 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 639 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 640 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 641 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 642 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 643 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 644 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 645 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 646 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 647 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 648 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 649 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 650 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 651 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 652 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 653 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 654 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 655 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 656 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 657 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 658 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 659 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 660 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 661 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 662 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 663 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 664 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 665 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 666 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 667 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 668 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 669 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 670 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 671 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 672 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 673 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 674 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 675 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 676 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 677 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 678 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 679 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 680 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 681 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 682 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 683 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 684 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 685 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 686 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 687 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 688 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 689 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 690 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 691 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 692 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 693 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 694 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 695 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggffail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 696 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 697 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 698 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 699 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 700 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 701 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 702 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 703 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 704 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 705 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 706 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 707 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 708 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 709 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 710 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 711 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 712 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 713 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 714 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 715 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 716 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 717 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 718 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 719 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 720 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 721 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 722 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 723 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 724 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 725 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 726 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 727 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 728 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 729 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 730 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 731 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 732 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 733 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 734 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 735 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 736 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 737 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 738 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 739 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 740 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 741 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 742 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 743 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 744 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 745 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 746 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 747 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 748 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 749 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 750 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 751 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 752 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 753 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 754 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 755 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 756 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 757 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 758 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 759 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 760 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 761 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 762 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 763 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 764 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin1ggfpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 765 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 766 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 767 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 768 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 769 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 770 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 771 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 772 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 773 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 774 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 775 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 776 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 777 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 778 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 779 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 780 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 781 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 782 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 783 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 784 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 785 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 786 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 787 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 788 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 789 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 790 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 791 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 792 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 793 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 794 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 795 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 796 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 797 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 798 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 799 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 800 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 801 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 802 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 803 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 804 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 805 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 806 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 807 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 808 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 809 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 810 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 811 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 812 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 813 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 814 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 815 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 816 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 817 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 818 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 819 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 820 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 821 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 822 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 823 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 824 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 825 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 826 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 827 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 828 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 829 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 830 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 831 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 832 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 833 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 834 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 835 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 836 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 837 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 838 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggffail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 839 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 840 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 841 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 842 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 843 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 844 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 845 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 846 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 847 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 848 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 849 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 850 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 851 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 852 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 853 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 854 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 855 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 856 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 857 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 858 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 859 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 860 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 861 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 862 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 863 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 864 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 865 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 866 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 867 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 868 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 869 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 870 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 871 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 872 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 873 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 874 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 875 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 876 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 877 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 878 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 879 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 880 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 881 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 882 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 883 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 884 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 885 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 886 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 887 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 888 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 889 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 890 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 891 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 892 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 893 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 894 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 895 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 896 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 897 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 898 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 899 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 900 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 901 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 902 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 903 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 904 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 905 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 906 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin2ggfpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 907 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 908 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 909 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 910 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 911 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 912 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 913 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 914 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 915 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 916 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 917 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 918 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 919 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 920 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 921 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 922 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 923 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 924 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 925 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 926 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 927 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 928 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 929 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 930 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 931 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 932 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 933 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 934 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 935 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 936 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 937 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 938 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 939 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 940 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 941 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 942 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 943 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 944 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 945 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 946 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 947 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 948 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 949 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 950 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 951 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 952 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 953 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 954 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 955 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 956 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 957 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 958 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 959 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 960 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 961 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 962 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 963 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 964 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 965 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 966 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 967 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 968 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 969 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 970 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 971 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 972 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 973 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 974 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ggF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ggF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 975 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 976 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 977 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 978 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 979 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggffail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 980 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 981 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 982 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 983 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 984 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 985 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 986 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 987 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 988 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 989 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 990 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 991 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 992 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 993 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 994 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 995 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 996 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 997 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 998 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 999 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1000 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1001 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1002 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1003 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1004 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1005 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1006 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1007 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1008 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1009 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1010 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1011 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1012 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1013 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1014 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1015 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1016 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1017 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1018 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1019 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1020 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1021 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1022 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1023 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1024 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1025 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1026 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1027 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1028 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1029 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1030 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1031 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1032 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1033 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1034 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1035 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1036 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1037 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1038 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1039 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1040 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1041 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1042 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1043 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1044 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1045 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1046 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin3ggfpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1047 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1048 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1049 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1050 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1051 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1052 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1053 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1054 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1055 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1056 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1057 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1058 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1059 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1060 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1061 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1062 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1063 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1064 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1065 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1066 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1067 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1068 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1069 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1070 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1071 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1072 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1073 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1074 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1075 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1076 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1077 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1078 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1079 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1080 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1081 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1082 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1083 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1084 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1085 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1086 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1087 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1088 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1089 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1090 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1091 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1092 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1093 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ggF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_ggF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1094 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1095 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1096 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1097 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1098 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1099 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1100 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1101 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1102 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1103 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1104 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1105 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1106 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1107 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1108 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1109 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1110 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1111 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1112 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1113 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1114 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1115 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggffail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1116 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1117 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1118 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1119 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1120 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1121 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1122 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1123 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1124 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1125 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1126 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1127 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1128 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1129 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1130 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1131 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1132 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1133 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1134 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1135 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1136 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1137 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1138 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1139 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1140 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1141 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1142 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1143 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1144 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1145 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1146 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1147 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1148 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1149 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1150 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1151 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1152 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1153 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1154 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1155 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1156 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1157 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1158 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1159 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1160 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1161 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1162 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1163 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1164 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1165 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1166 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1167 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1168 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1169 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1170 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1171 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1172 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1173 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1174 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1175 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1176 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1177 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1178 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1179 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_ggF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_ggF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1180 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1181 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1182 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1183 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin4ggfpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1184 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1185 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1186 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1187 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1188 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1189 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1190 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1191 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1192 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1193 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1194 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1195 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1196 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1197 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1198 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1199 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1200 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1201 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1202 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1203 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1204 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1205 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1206 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1207 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1208 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1209 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1210 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1211 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1212 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1213 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1214 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1215 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1216 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1217 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1218 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1219 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1220 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1221 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1222 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1223 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1224 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1225 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1226 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1227 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1228 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1229 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1230 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1231 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1232 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1233 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1234 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1235 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1236 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1237 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1238 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1239 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1240 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1241 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1242 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1243 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1244 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1245 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1246 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggffail2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1247 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1248 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1249 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1250 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1251 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1252 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1253 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1254 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1255 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1256 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1257 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1258 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1259 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1260 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1261 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1262 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016APV_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016APV_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1263 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1264 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1265 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1266 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1267 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1268 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1269 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1270 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1271 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1272 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1273 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1274 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1275 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1276 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1277 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2016_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2016_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1278 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1279 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1280 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1281 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1282 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1283 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1284 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1285 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1286 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1287 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1288 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1289 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1290 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1291 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1292 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1293 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1294 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1295 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1296 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1297 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_VBF_s0_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_VBF_s0_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1298 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_VBF_s1_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_VBF_s1_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1299 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_VBF_s2_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_VBF_s2_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1300 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_VV_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1301 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_WH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1302 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1303 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_ZH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1304 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1305 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1306 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_ggF_s3_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_ggF_s3_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1307 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_singlet_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1308 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_ttH_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1309 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2018_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P ptbin5ggfpass2018_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1310 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2016APV_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2016APV_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1311 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2016APV_deco1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2016APV_deco1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1312 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2016APV_deco2 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2016APV_deco2 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1313 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2016APV_deco3 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2016APV_deco3 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1314 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2016_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2016_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1315 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2016_deco1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2016_deco1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1316 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2017_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2017_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1317 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2017_deco1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2017_deco1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1318 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2018_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2018_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1319 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2018_deco1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_ggf2018_deco1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1320 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbfhi2016APV_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbfhi2016APV_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1321 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbfhi2016_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbfhi2016_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1322 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbfhi2017_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbfhi2017_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1323 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbfhi2018_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbfhi2018_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1324 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbflo2016APV_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbflo2016APV_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1325 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbflo2016_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbflo2016_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1326 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbflo2017_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbflo2017_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1327 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbflo2018_deco0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_MCtempl_vbflo2018_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1328 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016APVggf_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016APVggf_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1329 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016APVggf_pt_par0_rho_par1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016APVggf_pt_par0_rho_par1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1330 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016APVggf_pt_par0_rho_par2 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016APVggf_pt_par0_rho_par2 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1331 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016APVvbfhi_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016APVvbfhi_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1332 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016APVvbflo_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016APVvbflo_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1333 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016ggf_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016ggf_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1334 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016ggf_pt_par0_rho_par1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016ggf_pt_par0_rho_par1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1335 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016ggf_pt_par0_rho_par2 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016ggf_pt_par0_rho_par2 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1336 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016vbfhi_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016vbfhi_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1337 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2016vbflo_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2016vbflo_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1338 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017ggf_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2017ggf_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1339 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017ggf_pt_par0_rho_par1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2017ggf_pt_par0_rho_par1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1340 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017ggf_pt_par0_rho_par2 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2017ggf_pt_par0_rho_par2 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1341 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017vbfhi_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2017vbfhi_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1342 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017vbflo_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2017vbflo_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1343 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2018ggf_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2018ggf_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1344 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2018ggf_pt_par0_rho_par1 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2018ggf_pt_par0_rho_par1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1345 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2018ggf_pt_par0_rho_par2 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2018ggf_pt_par0_rho_par2 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1346 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2018vbfhi_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2018vbfhi_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1347 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2018vbflo_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tf_dataResidual_2018vbflo_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1348 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqeffSF_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqeffSF_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1349 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqeffSF_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqeffSF_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1350 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqeffSF_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqeffSF_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1351 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqeffSF_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqeffSF_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1352 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqnormSF_2016 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqnormSF_2016 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1353 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqnormSF_2016APV --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqnormSF_2016APV --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1354 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqnormSF_2017 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqnormSF_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi
if [ $1 -eq 1355 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqnormSF_2018 --algo impact --redefineSignalPOIs rVBF1000to1500,rggF300to450,rggF450to650,rggF650plus,rVBF1500plus -P tqqnormSF_2018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel/model_combined.root
fi