#!/usr/bin/env python
import uproot 
import numpy as np
from coffea.lookup_tools import dense_lookup
from coffea.lookup_tools import extractor

corrections = {}

pu = {}
pu["2016"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2016-preVFP-69200ub-99bins.root",
              "up": "data/PileupHistogram-goldenJSON-13tev-2016-preVFP-72400ub-99bins.root",
              "down": "data/PileupHistogram-goldenJSON-13tev-2016-preVFP-66000ub-99bins.root",
          }
pu["2017"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root",
              "up": "data/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root",
              "down": "data/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root",
          }
pu["2018"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root",
              "up": "data/PileupHistogram-goldenJSON-13tev-2018-72400ub-99bins.root",
              "down": "data/PileupHistogram-goldenJSON-13tev-2018-66000ub-99bins.root"
          }
              
pileup_corr = {}
norm = lambda x: x / x.sum()
for year,pdict in pu.items():
    pileup_corr[year] = {}
    data_pu = {}
    data_pu_edges = {}
    for var,pfile in pdict.items():
        with uproot.open(pfile) as ifile:
            data_pu[var] = norm(ifile["pileup"].values())
            data_pu_edges[var] = ifile["pileup"].axis().edges()

    # from mix.input.nbPileupEvents.probValue:
    # https://github.com/cms-sw/cmssw/blob/master/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py
    mc_pu = np.array([1.1840841518e-05, 3.46661037703e-05, 8.98772521472e-05, 7.47400487733e-05, 0.000123005176624,
                      0.000156501700614, 0.000154660478659, 0.000177496185603, 0.000324149805611, 0.000737524009713,
                      0.00140432980253, 0.00244424508696, 0.00380027898037, 0.00541093042612, 0.00768803501793,
                      0.010828224552, 0.0146608623707, 0.01887739113, 0.0228418813823, 0.0264817796874,
                      0.0294637401336, 0.0317960986171, 0.0336645950831, 0.0352638818387, 0.036869429333,
                      0.0382797316998, 0.039386705577, 0.0398389681346, 0.039646211131, 0.0388392805703,
                      0.0374195678161, 0.0355377892706, 0.0333383902828, 0.0308286549265, 0.0282914440969,
                      0.0257860718304, 0.02341635055, 0.0213126338243, 0.0195035612803, 0.0181079838989,
                      0.0171991315458, 0.0166377598339, 0.0166445341361, 0.0171943735369, 0.0181980997278,
                      0.0191339792146, 0.0198518804356, 0.0199714909193, 0.0194616474094, 0.0178626975229,
                      0.0153296785464, 0.0126789254325, 0.0100766041988, 0.00773867100481, 0.00592386091874,
                      0.00434706240169, 0.00310217013427, 0.00213213401899, 0.0013996000761, 0.000879148859271,
                      0.000540866009427, 0.000326115560156, 0.000193965828516, 0.000114607606623, 6.74262828734e-05,
                      3.97805301078e-05, 2.19948704638e-05, 9.72007976207e-06, 4.26179259146e-06, 2.80015581327e-06,
                      1.14675436465e-06, 2.52452411995e-07, 9.08394910044e-08, 1.14291987912e-08, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0])
    mask = mc_pu > 0.
    for var in data_pu.keys():
        corr = data_pu[var].copy()
        corr[mask] /= mc_pu[mask]
        pileup_corr[year][var] = dense_lookup.dense_lookup(corr,data_pu_edges[var])

for year in pileup_corr.keys():
    pileup_corr[year]["central"]._values = np.minimum(5,pileup_corr[year]["central"]._values)
    pileup_corr[year]["up"]._values = np.minimum(5,pileup_corr[year]["up"]._values)
    pileup_corr[year]["down"]._values = np.minimum(5,pileup_corr[year]["down"]._values)

    corrections['%s_pileupweight'%year] = pileup_corr[year]["central"]
    corrections['%s_pileupweight_puUp'%year] = pileup_corr[year]["up"]
    corrections['%s_pileupweight_puDown'%year] = pileup_corr[year]["down"]

# TODO: Update to UL
"""
Lepton ID, Isolation and Trigger SFs
Key: [ROOT file,
      Histogram label,
      Error Histogram label,
      Option]
Option: 0 for eta-pt, 1 for abseta-pt, 2 for pt-abseta
"""
lepton_sf_dict = {
    "elec_RECO":["egammaEffi_txt_EGM2D_runBCDEF_passingRECO.root","EGamma_SF2D","EGamma_SF2D_error",0],
    "elec_ID":["egammaEffi_txt_EGM2D_runBCDEF_passingID.root","EGamma_SF2D","EGamma_SF2D_error",0],
    "elec_TRIG32":["egammaEffi_txt_runBCDEF_passingEle32.root","EGamma_SF2D","EGamma_SF2D_error",0],
    "elec_TRIG115":["egammaEffi_txt_runBCDEF_passingEle115.json","Ele115_PtEtaBins/abseta_pt_SF_value","Ele115_PtEtaBins/abseta_pt_SF_error",1],
    "muon_ISO":["muonEff_RunBCDEF_SF_ISO.json","NUM_LooseRelIso_DEN_MediumID/abseta_pt_value","NUM_LooseRelIso_DEN_MediumID/abseta_pt_error",1],
    "muon_ID":["muonEff_RunBCDEF_SF_ID.json","NUM_MediumID_DEN_genTracks/abseta_pt_value","NUM_MediumID_DEN_genTracks/abseta_pt_error",1],
    "muon_TRIG27":["muonEff_RunBCDEF_SF_Trig_Nov17Nov2017.json","IsoMu27_PtEtaBins/pt_abseta_ratio_value","IsoMu27_PtEtaBins/pt_abseta_ratio_error",2],
    "muon_TRIG50":["muonEff_RunBCDEF_SF_Trig_Nov17Nov2017.json","Mu50_PtEtaBins/pt_abseta_ratio_value","Mu50_PtEtaBins/pt_abseta_ratio_error",2],
}
extractor = extractor()
for sfname, sfopts in lepton_sf_dict.items():
    extractor.add_weight_sets(["%s_value %s data/%s"%(sfname,sfopts[1],sfopts[0])])
    extractor.add_weight_sets(["%s_error %s data/%s"%(sfname,sfopts[2],sfopts[0])])
extractor.finalize()
evaluator = extractor.make_evaluator()

for sfname, sfopts in lepton_sf_dict.items():
    corrections["%s_value"%sfname] = evaluator["%s_value"%sfname]
    corrections["%s_error"%sfname] = evaluator["%s_error"%sfname]

import pickle
import gzip
with gzip.open('data/corrections.pkl.gz', 'wb') as f:
    pickle.dump(corrections, f, -1)
