#!/usr/bin/env python
import uproot 
import numpy as np
from coffea.lookup_tools import dense_lookup
from coffea.lookup_tools import extractor

corrections = {}

pu = {}
pu["2016APV"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2016-preVFP-69200ub-99bins.root",
                    "up": "data/PileupHistogram-goldenJSON-13tev-2016-preVFP-72400ub-99bins.root",
                    "down": "data/PileupHistogram-goldenJSON-13tev-2016-preVFP-66000ub-99bins.root",
                }
pu["2016"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2016-postVFP-69200ub-99bins.root",
                     "up": "data/PileupHistogram-goldenJSON-13tev-2016-postVFP-72400ub-99bins.root",
                     "down": "data/PileupHistogram-goldenJSON-13tev-2016-postVFP-66000ub-99bins.root",
                 }
pu["2017"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root",
              "up": "data/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root",
              "down": "data/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root",
          }
pu["2018"] = {"central": "data/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root",
              "up": "data/PileupHistogram-goldenJSON-13tev-2018-72400ub-99bins.root",
              "down": "data/PileupHistogram-goldenJSON-13tev-2018-66000ub-99bins.root"
          }

pu_mc = {}
pu_mc['2016APV'] = "data/pileup_mc_2016preVFP.root"
pu_mc['2016'] = "data/pileup_mc_2016postVFP.root"
pu_mc['2017'] = "data/pileup_mc_2017.root"
pu_mc['2018'] = "data/pileup_mc_2018.root"

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

    mc_pu = {}
    with uproot.open(pu_mc[year]) as ifile:
        mc_pu = norm(ifile["pileup"].values())
    # mc hist goes from -0.5 to 99.5, data hist goes from 0 to 99, data[i] -> mc[i-1] so I insert 0 at the front and delete the extra 0s at the end to match them up
    # adapted from https://hypernews.cern.ch/HyperNews/CMS/get/physics-validation/3689/1/1.html
    mc_pu = np.insert(mc_pu,0,0.)
    mc_pu = mc_pu[:-2]
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

import pickle
import gzip
with gzip.open('data/corrections.pkl.gz', 'wb') as f:
    pickle.dump(corrections, f, -1)
