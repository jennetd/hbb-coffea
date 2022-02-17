#!/usr/bin/env python
import json
import gzip
import uproot
import numexpr
import numpy as np
from coffea import hist, lookup_tools
from coffea.util import load, save
from coffea.hist import plot


corrections = {}

extractor = lookup_tools.extractor()
extractor.add_weight_sets(["2016_n2ddt_ * correction_files/n2ddt_transform_2016MC.root"])
extractor.add_weight_sets(["2017_n2ddt_ * correction_files/n2ddt_transform_2017MC.root"])
extractor.add_weight_sets(["2018_n2ddt_ * correction_files/n2ddt_transform_2018MC.root"])

extractor.add_weight_sets(["2016BF_mutrigger_ * correction_files/Muon2016_TriggerEfficienciesAndSF_RunBtoF.root"])
extractor.add_weight_sets(["2016GH_mutrigger_ * correction_files/Muon2016_TriggerEfficienciesAndSF_RunGH.root"])
extractor.add_weight_sets(["2017_mutrigger_ * correction_files/EfficienciesAndSF_RunBtoF_Nov17Nov2017.root"])
extractor.add_weight_sets(["2018_mutrigger_ * correction_files/Muon2018_RunABCD_AfterHLTUpdate_SF_trig.root"])

extractor.add_weight_sets(["2016BF_muid_ * correction_files/Muon2016_EfficienciesAndSF_RunBtoF.root"])
extractor.add_weight_sets(["2016GH_muid_ * correction_files/Muon2016_EfficienciesAndSF_RunGH.root"])
extractor.add_weight_sets(["2017_muid_ * correction_files/Muon2017_RunBCDEF_SF_ID.json"])
extractor.add_weight_sets(["2018_muid_ * correction_files/Muon2018_RunABCD_SF_ID.json"])

extractor.add_weight_sets(["2016BF_muiso_ * correction_files/Muon2016_IsoEfficienciesAndSF_RunBtoF.root"])
extractor.add_weight_sets(["2016GH_muiso_ * correction_files/Muon2016_IsoEfficienciesAndSF_RunGH.root"])
extractor.add_weight_sets(["2017_muiso_ * correction_files/Muon2017_RunBCDEF_SF_ISO.json"])
extractor.add_weight_sets(["2018_muiso_ * correction_files/Muon2018_RunABCD_SF_ISO.json"])

extractor.add_weight_sets(["W_qcd_kfactors * correction_files/WJets_QCD_NLO.root"])
extractor.add_weight_sets(["Z_qcd_kfactors * correction_files/ZJets_QCD_NLO.root"])

extractor.finalize()
evaluator = extractor.make_evaluator()

corrections['2016_n2ddt_rho_pt'] = evaluator['2016_n2ddt_Rho2D']
corrections['2017_n2ddt_rho_pt'] = evaluator['2017_n2ddt_Rho2D']
corrections['2018_n2ddt_rho_pt'] = evaluator['2018_n2ddt_Rho2D']

lumiw2016_GH = 16.146 / (16.146 + 19.721)
lumiw2016_BCDEF = 19.721 / (16.146 + 19.721)

corrections['2016_mutrigweight_pt_abseta'] = evaluator['2016BF_mutrigger_Mu50_OR_TkMu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA']
corrections['2016_mutrigweight_pt_abseta']._values *= lumiw2016_BCDEF
corrections['2016_mutrigweight_pt_abseta']._values += evaluator['2016GH_mutrigger_Mu50_OR_TkMu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA']._values * lumiw2016_GH
corrections['2016_mutrigweight_pt_abseta_mutrigweightShift'] = evaluator['2016BF_mutrigger_Mu50_OR_TkMu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA_error']
bf = evaluator['2016BF_mutrigger_Mu50_OR_TkMu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA_error']._values
gh = evaluator['2016GH_mutrigger_Mu50_OR_TkMu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA_error']._values
corrections['2016_mutrigweight_pt_abseta_mutrigweightShift']._values = np.hypot(bf*lumiw2016_BCDEF, gh*lumiw2016_GH)
corrections['2017_mutrigweight_pt_abseta'] = evaluator['2017_mutrigger_Mu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA']
corrections['2017_mutrigweight_pt_abseta_mutrigweightShift'] = evaluator['2017_mutrigger_Mu50_PtEtaBins/efficienciesDATA/pt_abseta_DATA_error']
corrections['2018_mutrigweight_pt_abseta'] = evaluator['2018_mutrigger_Mu50_OR_OldMu100_OR_TkMu100_PtEtaBins/efficienciesDATA/pt_abseta_DATA']
corrections['2018_mutrigweight_pt_abseta_mutrigweightShift'] = evaluator['2018_mutrigger_Mu50_OR_OldMu100_OR_TkMu100_PtEtaBins/efficienciesDATA/pt_abseta_DATA_error']

corrections['2016_muidweight_abseta_pt'] = evaluator['2016BF_muid_MC_NUM_LooseID_DEN_genTracks_PAR_pt_eta/efficienciesDATA/abseta_pt_DATA']
corrections['2016_muidweight_abseta_pt']._values *= lumiw2016_BCDEF
corrections['2016_muidweight_abseta_pt']._values += evaluator['2016GH_muid_MC_NUM_LooseID_DEN_genTracks_PAR_pt_eta/efficienciesDATA/abseta_pt_DATA']._values * lumiw2016_GH
corrections['2016_muidweight_abseta_pt_muidweightShift'] = evaluator['2016BF_muid_MC_NUM_LooseID_DEN_genTracks_PAR_pt_eta/efficienciesDATA/abseta_pt_DATA_error']
bf = evaluator['2016BF_muid_MC_NUM_LooseID_DEN_genTracks_PAR_pt_eta/efficienciesDATA/abseta_pt_DATA_error']._values
gh = evaluator['2016GH_muid_MC_NUM_LooseID_DEN_genTracks_PAR_pt_eta/efficienciesDATA/abseta_pt_DATA_error']._values
corrections['2016_muidweight_abseta_pt_muidweightShift']._values = np.hypot(bf*lumiw2016_BCDEF, gh*lumiw2016_GH)
corrections['2017_muidweight_abseta_pt'] = evaluator['2017_muid_NUM_LooseID_DEN_genTracks/abseta_pt_value']
corrections['2017_muidweight_abseta_pt_muidweightShift'] = evaluator['2017_muid_NUM_LooseID_DEN_genTracks/abseta_pt_error']
corrections['2018_muidweight_abseta_pt'] = evaluator['2018_muid_NUM_LooseID_DEN_genTracks/abseta_pt_value']
corrections['2018_muidweight_abseta_pt_muidweightShift'] = evaluator['2018_muid_NUM_LooseID_DEN_genTracks/abseta_pt_error']

corrections['2016_muisoweight_abseta_pt'] = evaluator['2016BF_muiso_LooseISO_LooseID_pt_eta/efficienciesDATA/abseta_pt_DATA']
corrections['2016_muisoweight_abseta_pt']._values *= lumiw2016_BCDEF
corrections['2016_muisoweight_abseta_pt']._values += evaluator['2016GH_muiso_LooseISO_LooseID_pt_eta/efficienciesDATA/abseta_pt_DATA']._values * lumiw2016_GH
corrections['2016_muisoweight_abseta_pt_muisoweightShift'] = evaluator['2016BF_muiso_LooseISO_LooseID_pt_eta/efficienciesDATA/abseta_pt_DATA_error']
bf = evaluator['2016BF_muiso_LooseISO_LooseID_pt_eta/efficienciesDATA/abseta_pt_DATA_error']._values
gh = evaluator['2016GH_muiso_LooseISO_LooseID_pt_eta/efficienciesDATA/abseta_pt_DATA_error']._values
corrections['2016_muisoweight_abseta_pt_muisoweightShift']._values = np.hypot(bf*lumiw2016_BCDEF, gh*lumiw2016_GH)
corrections['2017_muisoweight_abseta_pt'] = evaluator['2017_muiso_NUM_LooseRelIso_DEN_LooseID/abseta_pt_value']
corrections['2017_muisoweight_abseta_pt_muisoweightShift'] = evaluator['2017_muiso_NUM_LooseRelIso_DEN_LooseID/abseta_pt_error']
corrections['2018_muisoweight_abseta_pt'] = evaluator['2018_muiso_NUM_LooseRelIso_DEN_LooseID/abseta_pt_value']
corrections['2018_muisoweight_abseta_pt_muisoweightShift'] = evaluator['2018_muiso_NUM_LooseRelIso_DEN_LooseID/abseta_pt_error']


gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
cpar = np.array([1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17])
fpar = np.array([1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17])
def msd_weight(pt, eta):
    genw = gpar[0] + gpar[1]*np.power(pt*gpar[2], -gpar[3])
    ptpow = np.power.outer(pt, np.arange(cpar.size))
    cenweight = np.dot(ptpow, cpar)
    forweight = np.dot(ptpow, fpar)
    weight = np.where(np.abs(eta)<1.3, cenweight, forweight)
    return genw*weight

corrections['msdweight'] = msd_weight

with uproot.open("correction_files/kfactors.root") as kfactors:
    ewkW_num = kfactors['EWKcorr/W']
    ewkZ_num = kfactors['EWKcorr/Z']
    # ewkW_denom = kfactors['WJets_LO/inv_pt']
    # ewkZ_denom = kfactors['ZJets_LO/inv_pt']
    ewkW_denom = kfactors['WJets_012j_NLO/nominal']
    ewkZ_denom = kfactors['ZJets_012j_NLO/nominal']

edges = ewkW_num.edges
assert(all(np.array_equal(edges, h.edges) for h in [ewkW_denom, ewkZ_num, ewkZ_denom]))
ptrange = slice(np.searchsorted(edges, 250.), np.searchsorted(edges, 1000.) + 1)
corrections['W_nlo_over_lo_ewk'] = lookup_tools.dense_lookup.dense_lookup(ewkW_num.values[ptrange] / ewkW_denom.values[ptrange], edges[ptrange])
corrections['Z_nlo_over_lo_ewk'] = lookup_tools.dense_lookup.dense_lookup(ewkZ_num.values[ptrange] / ewkZ_denom.values[ptrange], edges[ptrange])

with uproot.open("correction_files/WJets_QCD_NLO.root") as kfactors:
    qcdW_2016_nlo = kfactors['W_NLO_QCD_2016']
    qcdW_2017_nlo = kfactors['W_NLO_QCD_2017']

with uproot.open("correction_files/ZJets_QCD_NLO.root") as kfactors:
    qcdZ_2016_nlo = kfactors['Z_NLO_QCD_2016']
    qcdZ_2017_nlo = kfactors['Z_NLO_QCD_2017']

edges = qcdW_2016_nlo.edges
assert(all(np.array_equal(edges, h.edges) for h in [qcdW_2017_nlo, qcdZ_2016_nlo, qcdZ_2017_nlo]))
ptrange = slice(np.searchsorted(edges, 250.), np.searchsorted(edges, 1000.) + 1)

corrections['2016_W_nlo_qcd'] = lookup_tools.dense_lookup.dense_lookup(qcdW_2016_nlo.values[ptrange], edges[ptrange])
corrections['2017_W_nlo_qcd'] = lookup_tools.dense_lookup.dense_lookup(qcdW_2017_nlo.values[ptrange], edges[ptrange])
corrections['2016_Z_nlo_qcd'] = lookup_tools.dense_lookup.dense_lookup(qcdZ_2016_nlo.values[ptrange], edges[ptrange])
corrections['2017_Z_nlo_qcd'] = lookup_tools.dense_lookup.dense_lookup(qcdZ_2017_nlo.values[ptrange], edges[ptrange])


with uproot.open("correction_files/pileUp_Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.root") as fin_pileup:
    norm = lambda x: x / x.sum()
    data_pu = norm(fin_pileup["pileup"].values)
    data_pu_puUp = norm(fin_pileup["pileup_plus"].values)
    data_pu_puDown = norm(fin_pileup["pileup_minus"].values)

    # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_2016_25ns_Moriond17MC_PoissonOOTPU_cfi.py
    mc_pu = np.array([
            1.78653e-05 ,2.56602e-05 ,5.27857e-05 ,8.88954e-05 ,0.000109362 ,0.000140973 ,0.000240998 ,0.00071209 ,
            0.00130121 ,0.00245255 ,0.00502589 ,0.00919534 ,0.0146697 ,0.0204126 ,0.0267586 ,0.0337697 ,0.0401478 ,
            0.0450159 ,0.0490577 ,0.0524855 ,0.0548159 ,0.0559937 ,0.0554468 ,0.0537687 ,0.0512055 ,0.0476713 ,
            0.0435312 ,0.0393107 ,0.0349812 ,0.0307413 ,0.0272425 ,0.0237115 ,0.0208329 ,0.0182459 ,0.0160712 ,
            0.0142498 ,0.012804 ,0.011571 ,0.010547 ,0.00959489 ,0.00891718 ,0.00829292 ,0.0076195 ,0.0069806 ,
            0.0062025 ,0.00546581 ,0.00484127 ,0.00407168 ,0.00337681 ,0.00269893 ,0.00212473 ,0.00160208 ,
            0.00117884 ,0.000859662 ,0.000569085 ,0.000365431 ,0.000243565 ,0.00015688 ,9.88128e-05 ,
            6.53783e-05 ,3.73924e-05 ,2.61382e-05 ,2.0307e-05 ,1.73032e-05 ,1.435e-05 ,1.36486e-05 ,1.35555e-05 ,
            1.37491e-05 ,1.34255e-05 ,1.33987e-05 ,1.34061e-05 ,1.34211e-05 ,1.34177e-05 ,1.32959e-05 ,1.33287e-05
    ])
    mc_pu = np.r_[mc_pu, np.zeros(25)]
    mask = mc_pu > 0.
    corr = data_pu.copy()
    corr_puUp = data_pu_puUp.copy()
    corr_puDown = data_pu_puDown.copy()
    corr[mask] /= mc_pu[mask]
    corr_puUp[mask] /= mc_pu[mask]
    corr_puDown[mask] /= mc_pu[mask]
    pileup_corr = lookup_tools.dense_lookup.dense_lookup(corr, fin_pileup["pileup"].edges)
    pileup_corr_puUp = lookup_tools.dense_lookup.dense_lookup(corr_puUp, fin_pileup["pileup"].edges)
    pileup_corr_puDown = lookup_tools.dense_lookup.dense_lookup(corr_puDown, fin_pileup["pileup"].edges)

corrections['2016_pileupweight'] = pileup_corr
corrections['2016_pileupweight_puUp'] = pileup_corr_puUp
corrections['2016_pileupweight_puDown'] = pileup_corr_puDown

pileup_corr = load('correction_files/pileup_mc.coffea')
del pileup_corr['data_obs_jet']
del pileup_corr['data_obs_mu']
with uproot.open("correction_files/pileup_Cert_294927-306462_13TeV_PromptReco_Collisions17_withVar.root") as fin_pileup:
    norm = lambda x: x / x.sum()
    data_pu = norm(fin_pileup["pileup"].values)
    data_pu_puUp = norm(fin_pileup["pileup_plus"].values)
    data_pu_puDown = norm(fin_pileup["pileup_minus"].values)

    pileup_corr_puUp = {}
    pileup_corr_puDown = {}
    for k in pileup_corr.keys():
        if pileup_corr[k].value.sum() == 0:
            print("sample has no MC pileup:", k)
            continue
        mc_pu = norm(pileup_corr[k].value)
        mask = mc_pu > 0.
        corr = data_pu.copy()
        corr_puUp = data_pu_puUp.copy()
        corr_puDown = data_pu_puDown.copy()
        corr[mask] /= mc_pu[mask]
        corr_puUp[mask] /= mc_pu[mask]
        corr_puDown[mask] /= mc_pu[mask]
        pileup_corr[k] = lookup_tools.dense_lookup.dense_lookup(corr, fin_pileup["pileup"].edges)
        pileup_corr_puUp[k] = lookup_tools.dense_lookup.dense_lookup(corr_puUp, fin_pileup["pileup"].edges)
        pileup_corr_puDown[k] = lookup_tools.dense_lookup.dense_lookup(corr_puDown, fin_pileup["pileup"].edges)

corrections['2017_pileupweight_dataset'] = pileup_corr
corrections['2017_pileupweight_dataset_puUp'] = pileup_corr_puUp
corrections['2017_pileupweight_dataset_puDown'] = pileup_corr_puDown

with uproot.open("correction_files/pileUp_Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.root") as fin_pileup:
    norm = lambda x: x / x.sum()
    data_pu = norm(fin_pileup["pileup"].values)
    data_pu_puUp = norm(fin_pileup["pileup_plus"].values)
    data_pu_puDown = norm(fin_pileup["pileup_minus"].values)

    # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_2018_25ns_JuneProjectionFull18_PoissonOOTPU_cfi.py
    mc_pu = np.array([
        4.695341e-10, 1.206213e-06, 1.162593e-06, 6.118058e-06, 1.626767e-05,
        3.508135e-05, 7.12608e-05, 0.0001400641, 0.0002663403, 0.0004867473,
        0.0008469, 0.001394142, 0.002169081, 0.003198514, 0.004491138,
        0.006036423, 0.007806509, 0.00976048, 0.0118498, 0.01402411,
        0.01623639, 0.01844593, 0.02061956, 0.02273221, 0.02476554,
        0.02670494, 0.02853662, 0.03024538, 0.03181323, 0.03321895,
        0.03443884, 0.035448, 0.03622242, 0.03674106, 0.0369877,
        0.03695224, 0.03663157, 0.03602986, 0.03515857, 0.03403612,
        0.0326868, 0.03113936, 0.02942582, 0.02757999, 0.02563551,
        0.02362497, 0.02158003, 0.01953143, 0.01750863, 0.01553934,
        0.01364905, 0.01186035, 0.01019246, 0.008660705, 0.007275915,
        0.006043917, 0.004965276, 0.004035611, 0.003246373, 0.002585932,
        0.002040746, 0.001596402, 0.001238498, 0.0009533139, 0.0007282885,
        0.000552306, 0.0004158005, 0.0003107302, 0.0002304612, 0.0001696012,
        0.0001238161, 8.96531e-05, 6.438087e-05, 4.585302e-05, 3.23949e-05,
        2.271048e-05, 1.580622e-05, 1.09286e-05, 7.512748e-06, 5.140304e-06,
        3.505254e-06, 2.386437e-06, 1.625859e-06, 1.111865e-06, 7.663272e-07,
        5.350694e-07, 3.808318e-07, 2.781785e-07, 2.098661e-07, 1.642811e-07,
        1.312835e-07, 1.081326e-07, 9.141993e-08, 7.890983e-08, 6.91468e-08,
        6.119019e-08, 5.443693e-08, 4.85036e-08, 4.31486e-08, 3.822112e-08
    ])
    mask = mc_pu > 0.
    corr = data_pu.copy()
    corr_puUp = data_pu_puUp.copy()
    corr_puDown = data_pu_puDown.copy()
    corr[mask] /= mc_pu[mask]
    corr_puUp[mask] /= mc_pu[mask]
    corr_puDown[mask] /= mc_pu[mask]
    pileup_corr = lookup_tools.dense_lookup.dense_lookup(corr, fin_pileup["pileup"].edges)
    pileup_corr_puUp = lookup_tools.dense_lookup.dense_lookup(corr_puUp, fin_pileup["pileup"].edges)
    pileup_corr_puDown = lookup_tools.dense_lookup.dense_lookup(corr_puDown, fin_pileup["pileup"].edges)

corrections['2018_pileupweight'] = pileup_corr
corrections['2018_pileupweight_puUp'] = pileup_corr_puUp
corrections['2018_pileupweight_puDown'] = pileup_corr_puDown


with uproot.open("correction_files/RUNTriggerEfficiencies_SingleMuon_Run2016_V2p1_v03.root") as fin:
    denom = fin["DijetTriggerEfficiencySeveralTriggers/jet1SoftDropMassjet1PtDenom_cutJet"]
    num = fin["DijetTriggerEfficiencySeveralTriggers/jet1SoftDropMassjet1PtPassing_cutJet"]
    eff = num.values/np.maximum(denom.values, 1)
    efferr = plot.clopper_pearson_interval(num.values, denom.values)
    msd_bins, pt_bins = num.edges
    # Cut pt < 200
    cutpt = pt_bins >= 200
    pt_bins = pt_bins[cutpt]
    cutpt = cutpt[:-1]
    eff = eff[:,cutpt]
    eff_trigweightDown = efferr[0,:,cutpt]
    eff_trigweightUp = efferr[1,:,cutpt]

corrections['2016_trigweight_msd_pt'] = lookup_tools.dense_lookup.dense_lookup(eff, (msd_bins, pt_bins))
corrections['2016_trigweight_msd_pt_trigweightDown'] = lookup_tools.dense_lookup.dense_lookup(eff_trigweightDown, (msd_bins, pt_bins))
corrections['2016_trigweight_msd_pt_trigweightUp'] = lookup_tools.dense_lookup.dense_lookup(eff_trigweightUp, (msd_bins, pt_bins))

with uproot.open("correction_files/TrigEff_2017BtoF_noPS_Feb21.root") as fin:
    denom = fin["h_denom"]
    num = fin["h_numer"]
    eff = num.values/np.maximum(denom.values, 1)
    efferr = plot.clopper_pearson_interval(num.values, denom.values)
    msd_bins, pt_bins = num.edges
    # Cut pt < 200
    pt_bins = pt_bins[8:]
    eff = eff[:,8:]
    eff_trigweightDown = efferr[0,:,8:]
    eff_trigweightUp = efferr[1,:,8:]

corrections['2017_trigweight_msd_pt'] = lookup_tools.dense_lookup.dense_lookup(eff, (msd_bins, pt_bins))
corrections['2017_trigweight_msd_pt_trigweightDown'] = lookup_tools.dense_lookup.dense_lookup(eff_trigweightDown, (msd_bins, pt_bins))
corrections['2017_trigweight_msd_pt_trigweightUp'] = lookup_tools.dense_lookup.dense_lookup(eff_trigweightUp, (msd_bins, pt_bins))

with uproot.open("correction_files/TrigEff_2018_Feb21.root") as fin:
    denom = fin["h_denom"]
    num = fin["h_numer"]
    eff = num.values/np.maximum(denom.values, 1)
    efferr = plot.clopper_pearson_interval(num.values, denom.values)
    msd_bins, pt_bins = num.edges
    # Cut pt < 200
    pt_bins = pt_bins[8:]
    eff = eff[:,8:]
    eff_trigweightDown = efferr[0,:,8:]
    eff_trigweightUp = efferr[1,:,8:]

corrections['2018_trigweight_msd_pt'] = lookup_tools.dense_lookup.dense_lookup(eff, (msd_bins, pt_bins))
corrections['2018_trigweight_msd_pt_trigweightDown'] = lookup_tools.dense_lookup.dense_lookup(eff_trigweightDown, (msd_bins, pt_bins))
corrections['2018_trigweight_msd_pt_trigweightUp'] = lookup_tools.dense_lookup.dense_lookup(eff_trigweightUp, (msd_bins, pt_bins))


with open("correction_files/TriggerBitMap.json") as fin:
    trigger_bitmap = json.load(fin)

def triggermask(names, triggerMap):
    version     = names['version']
    hltNames    = names['names']
    branchName  = names['branchName']
    if version in triggerMap:
        bits = triggerMap[version]
    else:
        raise ValueError("Cannot find triggerbit map of the requested bit version =%s. Possible versions are: %s" % (version, ",".join(triggerMap.keys())))
    tCuts = []
    mask = np.array(0, dtype='uint64')
    for hltName in hltNames:
        if hltName not in bits:
            raise ValueError("Cannot find the TriggerBit for %s" % hltName)
        mask |= np.array(1<<int(bits[hltName]), dtype=mask.dtype)
    return mask

triggerNames_2016 = {
    "version": "zprimebit-15.01",
    "branchName":"triggerBits",
    "names": [
        "HLT_PFHT800_v*",
        "HLT_PFHT900_v*",
        "HLT_AK8PFJet360_TrimMass30_v*",
        'HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v*',
        "HLT_PFHT650_WideJetMJJ950DEtaJJ1p5_v*",
        "HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v*",
        "HLT_AK8DiPFJet280_200_TrimMass30_BTagCSV_p20_v*" ,
        "HLT_PFJet450_v*",
    ]
}

corrections['2016_triggerMask'] = triggermask(triggerNames_2016, trigger_bitmap)

triggerNames_2017 = {
    "version": "zprimebit-15.01",
    "branchName":"triggerBits",
    "names": [
        "HLT_AK8PFJet330_PFAK8BTagCSV_p17_v*",
        "HLT_PFHT1050_v*",
        "HLT_AK8PFJet400_TrimMass30_v*",
        "HLT_AK8PFJet420_TrimMass30_v*",
        "HLT_AK8PFHT800_TrimMass50_v*",
        "HLT_PFJet500_v*",
        "HLT_AK8PFJet500_v*"
    ]
}

corrections['2017_triggerMask'] = triggermask(triggerNames_2017, trigger_bitmap)

triggerNames_2018 = {
    "version": "zprimebit-15.01",
    "branchName": "triggerBits",
    "names": [
        'HLT_AK8PFJet400_TrimMass30_v*',
        'HLT_AK8PFJet420_TrimMass30_v*',
        'HLT_AK8PFHT800_TrimMass50_v*',
        'HLT_PFHT1050_v*',
        'HLT_PFJet500_v*',
        'HLT_AK8PFJet500_v*',
        'HLT_AK8PFJet330_PFAK8BTagCSV_p17_v*',
        "HLT_AK8PFJet330_TrimMass30_PFAK8BoostedDoubleB_np4_v*",
    ],
}

corrections['2018_triggerMask'] = triggermask(triggerNames_2018, trigger_bitmap)


def read_xsections(filename):
    out = {}
    with open(filename) as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            dataset, xsexpr, *_ = line.split()
            try:
                xs = float(numexpr.evaluate(xsexpr))
            except:
                print("numexpr evaluation failed for line: %s" % line)
                raise
            if xs <= 0:
                warnings.warn("Cross section is <= 0 in line: %s" % line, RuntimeWarning)
            out[dataset] = xs
    return out

# curl -O https://raw.githubusercontent.com/kakwok/ZPrimePlusJet/newTF/analysis/ggH/xSections.dat
corrections['xsections'] = read_xsections("metadata/xSections.dat")

save(corrections, 'corrections.coffea')
