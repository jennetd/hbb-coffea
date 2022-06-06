import correctionlib
import awkward as ak
from coffea import processor, hist, util
import numpy as np
import pickle as pkl 
import importlib.resources

from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
btagWPs = {
    "deepJet": {
        '2016APV': {
            'L': 0.0508,
            'M': 0.2598,
            'T': 0.6502,
        },
        '2016': {
            'L': 0.0480,
            'M': 0.2489,
            'T': 0.6377,
        },
        '2017': {
            'L': 0.0532,
            'M': 0.3040,
            'T': 0.7476,
        },
        '2018': {
            'L': 0.0490,
            'M': 0.2783,
            'T': 0.7100,
        }
    }
}
taggerBranch = {
    "deepJet": "btagDeepFlavB",
#    "deepCSV": "btagDeep"
}

class BTagEfficiency(processor.ProcessorABC):
    def __init__(self, year='2017'):
        self._year = year
        self._accumulator = hist.Hist(
            'Events',
            hist.Cat('tagger', 'Tagger'),
            hist.Bin('passWP', 'passWP',2,0,2),
            hist.Bin('flavor', 'Jet hadronFlavour', [0, 4, 5]),
            hist.Bin('pt', 'Jet pT', 20, 40, 300),
            hist.Bin('abseta', 'Jet abseta', 4, 0, 2.5)
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
        ]

        out = self.accumulator.identity()
        tags = [
#            ('deepcsv', 'btagDeepB', 'M'),
            ('deepJet', 'btagDeepFlavB', 'M'),
        ]

        for tagger, branch, wp in tags:
            passbtag = jets[branch] > btagWPs[tagger][self._year][wp]

            out.fill(tagger=tagger,
                     pt=ak.flatten(jets.pt),
                     abseta=ak.flatten(abs(jets.eta)),
                     flavor=ak.flatten(jets.hadronFlavour),
                     passWP=ak.flatten(passbtag)
                 )
            
        return out

    def postprocess(self, a):
        return a

class BTagCorrector:
    def __init__(self, wp, tagger="deepJet", year="2017", mod=""):
        self._year = year+mod
        self._tagger = tagger
        self._wp = wp
        self._btagwp = btagWPs[tagger][year+mod][wp]
        self._branch = taggerBranch[tagger]

        # more docs at https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_btagging_Run2_UL/BTV_btagging_201*_UL.html   
        if year == '2016':
            self._cset = correctionlib.CorrectionSet.from_file("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz")
        elif year == '2016APV':
            self._cset = correctionlib.CorrectionSet.from_file("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz")
        else:
            self._cset = correctionlib.CorrectionSet.from_file(f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/{year}_UL/btagging.json.gz")

        # efficiency lookup
        self.efflookup = util.load(f'boostedhiggs/data/btageff_{tagger}_{wp}_{year}.coffea')

    def lighttagSF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset["%s_incl"%self._tagger].evaluate(syst, self._wp, np.array(j.hadronFlavour), np.array(abs(j.eta)), np.array(j.pt))
        return ak.unflatten(sf, nj)
        
    def btagSF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset["%s_comb"%self._tagger].evaluate(syst, self._wp, np.array(j.hadronFlavour), np.array(abs(j.eta)), np.array(j.pt))
        return ak.unflatten(sf, nj)

    def addBtagWeight(self, jets, weights):
        """
        Adding one common multiplicative SF (including bcjets + lightjets)
        weights: weights class from coffea
        jets: jets selected in your analysis
        """
        
        lightJets = jets[(jets.hadronFlavour == 0) & (abs(jets.eta)<2.5)]
        bcJets = jets[(jets.hadronFlavour > 0) & (abs(jets.eta)<2.5)]

        lightEff = self.efflookup(lightJets.hadronFlavour, lightJets.pt, abs(lightJets.eta))
        bcEff = self.efflookup(bcJets.hadronFlavour, bcJets.pt, abs(bcJets.eta))

        lightPass = lightJets[self._branch] > self._btagwp
        bcPass = bcJets[self._branch] > self._btagwp

        def combine(eff, sf, passbtag):
            # tagged SF = SF*eff / eff = SF
            tagged_sf = ak.prod(sf[passbtag], axis=-1)
            # untagged SF = (1 - SF*eff) / (1 - eff)
            untagged_sf = ak.prod(((1 - sf*eff) / (1 - eff))[~passbtag], axis=-1)

            return ak.fill_none(tagged_sf * untagged_sf, 1.)
#            return tagged_sf * untagged_sf
            
        lightweight = combine(
            lightEff,
            self.lighttagSF(lightJets, "central"),
            lightPass
        )
        lightweight = np.minimum(5,lightweight)

        weights.add('btagLightSF', lightweight)
        print('Light = ',max(lightweight))
        bcweight = combine(
            bcEff,
            self.btagSF(bcJets, "central"),
            bcPass
        )
        bcweight = np.minimum(5,bcweight)

        weights.add('btagBCSF', bcweight)
        print('BC = ',max(bcweight))
        
        # nominal weight = btagSF (btagSFbc*btagSFlight)
        nominal = lightweight * bcweight
#        weights.add('btagSF', nominal )

        # systematics:
        # btagSFlight_{year}: btagSFlight_up/down
        # btagSFbc_{year}: btagSFbc_up/down
        # btagSFlight_correlated: btagSFlight_up/down_correlated
        # btagSFbc_correlated:  btagSFbc_up/down_correlated
        weights.add(
            'btagSFlight_%s'%self._year,
            np.ones(len(nominal)),
            weightUp=combine(
                lightEff,
                self.lighttagSF(lightJets, "up"),
                lightPass
            ),
            weightDown=combine(
                lightEff,
                self.lighttagSF(lightJets, "down"),
                lightPass
            )
        )
        weights.add(
            'btagSFbc_%s'%self._year, 
            np.ones(len(nominal)),
            weightUp=combine(
                bcEff,
                self.btagSF(bcJets, "up"),
                bcPass
            ),
            weightDown=combine(
                bcEff,
                self.btagSF(bcJets, "down"),
                bcPass
            )
        )
        weights.add(
            'btagSFlight_correlated',
            np.ones(len(nominal)),
            weightUp=combine(
                lightEff,
                self.lighttagSF(lightJets, "up_correlated"),
                lightPass
            ),
            weightDown=combine(
                lightEff,
                self.lighttagSF(lightJets, "down_correlated"),
                lightPass
            )
        )
        weights.add(
            'btagSFbc_correlated',
            np.ones(len(nominal)),
            weightUp=combine(
                bcEff,
                self.btagSF(bcJets, "up_correlated"),
                bcPass
            ),
            weightDown=combine(
                bcEff,
                self.btagSF(bcJets, "down_correlated"),
                bcPass
            )
        )
        return nominal
