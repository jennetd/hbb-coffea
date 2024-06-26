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
  combine -M MultiDimFit -n _paramFit_Test_CMS_L1Prefiring_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_L1Prefiring_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 1 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_eff_bb_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_eff_bb_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 2 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_PU_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_PU_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 3 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_W_d2kappa_EW --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_W_d2kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 4 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_W_d3kappa_EW --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_W_d3kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 5 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_Z_d2kappa_EW --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_Z_d2kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 6 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_Z_d3kappa_EW --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_Z_d3kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 7 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_btagSFbc_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 8 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFbc_correlated --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_btagSFbc_correlated --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 9 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_btagSFlight_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 10 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_btagSFlight_correlated --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_btagSFlight_correlated --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 11 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d1K_NLO --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_d1K_NLO --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 12 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d1kappa_EW --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_d1kappa_EW --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 13 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d2K_NLO --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_d2K_NLO --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 14 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_d3K_NLO --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_d3K_NLO --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 15 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_1_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_ddb_1_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 16 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_2_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_ddb_2_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 17 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_ddb_3_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_ddb_3_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 18 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_e_veto_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_e_veto_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 19 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_jet_trigger_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_jet_trigger_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 20 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_trigger_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_mu_trigger_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 21 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_mu_veto_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_mu_veto_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 22 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 23 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_1_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_pt_1_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 24 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_2_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_pt_2_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 25 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_3_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_pt_3_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 26 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_4_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_pt_4_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 27 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_5_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_pt_5_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 28 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_scale_pt_6_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_scale_pt_6_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 29 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_smear_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_smear_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 30 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_tau_veto_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_tau_veto_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 31 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_vbfmucr_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_vbfmucr_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 32 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_1_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_veff_1_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 33 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_veff_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 34 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_2_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_veff_2_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 35 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_hbb_veff_3_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_hbb_veff_3_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 36 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_lumi_13TeV_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 37 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_correlated --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_lumi_13TeV_correlated --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 38 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_lumi_13TeV_correlated_20172018 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_lumi_13TeV_correlated_20172018 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 39 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_id_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_mu_id_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 40 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_mu_iso_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_mu_iso_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 41 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_res_j_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_res_j_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 42 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_scale_j_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_scale_j_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 43 ]; then
  combine -M MultiDimFit -n _paramFit_Test_CMS_ues_j_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P CMS_ues_j_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 44 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_VBF --algo impact --redefineSignalPOIs rVBF,rggF -P QCDscale_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 45 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_VH --algo impact --redefineSignalPOIs rVBF,rggF -P QCDscale_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 46 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_ggF --algo impact --redefineSignalPOIs rVBF,rggF -P QCDscale_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 47 ]; then
  combine -M MultiDimFit -n _paramFit_Test_QCDscale_ttH --algo impact --redefineSignalPOIs rVBF,rggF -P QCDscale_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 48 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_VBF --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_FSR_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 49 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_VH --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_FSR_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 50 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_ggF --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_FSR_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 51 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_FSR_ttH --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_FSR_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 52 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_VBF --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_ISR_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 53 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_VH --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_ISR_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 54 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_ggF --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_ISR_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 55 ]; then
  combine -M MultiDimFit -n _paramFit_Test_UEPS_ISR_ttH --algo impact --redefineSignalPOIs rVBF,rggF -P UEPS_ISR_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 56 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_QCD_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRfail2017_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 57 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRfail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 58 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRfail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 59 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRfail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 60 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRfail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 61 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRfail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRfail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 62 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_QCD_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRpass2017_QCD_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 63 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 64 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 65 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 66 ]; then
  combine -M MultiDimFit -n _paramFit_Test_muonCRpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P muonCRpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 67 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_VBF --algo impact --redefineSignalPOIs rVBF,rggF -P pdf_Higgs_VBF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 68 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_VH --algo impact --redefineSignalPOIs rVBF,rggF -P pdf_Higgs_VH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 69 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_ggF --algo impact --redefineSignalPOIs rVBF,rggF -P pdf_Higgs_ggF --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 70 ]; then
  combine -M MultiDimFit -n _paramFit_Test_pdf_Higgs_ttH --algo impact --redefineSignalPOIs rVBF,rggF -P pdf_Higgs_ttH --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 71 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 72 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 73 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 74 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 75 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 76 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 77 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 78 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 79 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 80 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 81 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 82 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 83 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 84 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 85 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 86 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 87 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 88 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 89 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 90 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 91 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 92 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 93 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 94 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 95 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 96 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 97 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 98 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 99 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 100 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 101 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 102 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 103 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 104 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 105 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 106 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 107 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 108 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 109 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 110 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 111 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 112 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhifail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhifail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 113 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 114 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 115 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 116 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 117 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 118 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 119 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 120 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 121 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 122 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 123 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 124 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 125 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 126 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbfhipass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbfhipass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 127 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 128 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 129 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 130 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 131 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 132 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 133 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 134 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 135 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 136 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 137 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 138 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 139 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 140 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflofail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflofail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 141 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 142 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 143 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 144 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 145 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 146 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 147 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 148 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 149 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 150 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 151 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 152 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 153 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 154 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin0vbflopass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin0vbflopass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 155 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 156 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 157 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 158 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 159 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 160 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 161 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 162 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 163 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 164 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 165 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 166 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 167 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 168 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 169 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 170 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 171 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 172 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 173 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 174 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 175 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 176 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 177 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 178 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 179 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 180 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 181 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 182 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin1ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin1ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 183 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 184 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 185 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 186 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 187 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 188 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 189 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 190 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 191 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 192 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 193 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 194 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 195 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 196 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 197 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 198 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 199 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 200 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 201 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 202 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 203 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 204 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 205 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 206 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 207 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 208 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 209 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 210 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin2ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin2ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 211 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 212 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 213 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 214 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 215 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 216 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 217 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 218 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 219 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 220 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 221 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 222 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 223 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 224 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 225 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 226 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 227 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 228 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 229 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 230 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 231 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 232 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 233 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 234 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 235 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 236 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 237 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 238 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin3ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin3ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 239 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 240 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 241 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 242 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 243 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 244 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 245 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 246 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 247 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 248 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 249 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 250 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 251 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 252 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 253 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 254 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 255 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 256 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 257 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 258 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 259 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 260 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 261 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 262 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 263 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 264 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 265 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 266 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin4ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin4ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 267 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 268 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 269 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 270 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 271 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 272 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 273 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 274 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 275 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 276 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 277 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 278 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 279 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 280 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggffail2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggffail2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 281 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_EWKW_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_EWKW_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 282 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_EWKZ_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_EWKZ_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 283 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_EWKZbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_EWKZbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 284 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_VBF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_VBF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 285 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_VV_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_VV_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 286 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_WH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_WH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 287 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_Wjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_Wjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 288 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ZH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_ZH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 289 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_Zjets_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_Zjets_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 290 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_Zjetsbb_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_Zjetsbb_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 291 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ggF_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_ggF_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 292 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_singlet_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_singlet_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 293 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ttH_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_ttH_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 294 ]; then
  combine -M MultiDimFit -n _paramFit_Test_ptbin5ggfpass2017_ttbar_mcstat --algo impact --redefineSignalPOIs rVBF,rggF -P ptbin5ggfpass2017_ttbar_mcstat --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 295 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2017_deco0 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_MCtempl_ggf2017_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 296 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_ggf2017_deco1 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_MCtempl_ggf2017_deco1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 297 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbfhi2017_deco0 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_MCtempl_vbfhi2017_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 298 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_MCtempl_vbflo2017_deco0 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_MCtempl_vbflo2017_deco0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 299 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017ggf_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_dataResidual_2017ggf_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 300 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017ggf_pt_par0_rho_par1 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_dataResidual_2017ggf_pt_par0_rho_par1 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 301 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017ggf_pt_par0_rho_par2 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_dataResidual_2017ggf_pt_par0_rho_par2 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 302 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017vbfhi_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_dataResidual_2017vbfhi_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 303 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tf_dataResidual_2017vbflo_pt_par0_rho_par0 --algo impact --redefineSignalPOIs rVBF,rggF -P tf_dataResidual_2017vbflo_pt_par0_rho_par0 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 304 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqeffSF_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P tqqeffSF_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi
if [ $1 -eq 305 ]; then
  combine -M MultiDimFit -n _paramFit_Test_tqqnormSF_2017 --algo impact --redefineSignalPOIs rVBF,rggF -P tqqnormSF_2017 --floatOtherPOIs 1 --saveInactivePOI 1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 -m 125 -d output/testModel_2017/model_combined.root --setParameters rVBF=1,rggF=1
fi