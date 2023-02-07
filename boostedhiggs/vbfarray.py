import logging
import numpy as np
import awkward as ak
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
import copy
from collections import defaultdict
from coffea import processor, hist
import hist as hist2
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from boostedhiggs.btag import BTagCorrector
from boostedhiggs.common import (
    getBosons,
    bosonFlavor,
    pass_json_array,
)
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    powheg_to_nnlops,
    add_pileup_weight,
    add_VJets_kFactors,
    add_jetTriggerSF,
    add_muonSFs,
    jet_factory,
    fatjet_factory,
    add_jec_variables,
    met_factory,
    lumiMasks,
)


logger = logging.getLogger(__name__)

def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

def pad_val(
    arr: ak.Array,
    value: float,
    target: int = None,
    axis: int = 0,
    to_numpy: bool = False,
    clip: bool = True):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    if target:
        ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=None)
    else:
        ret = ak.fill_none(arr, value, axis=None)
    return ret.to_numpy() if to_numpy else ret

class VBFArrayProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt', tagger='v2',
                 nnlops_rew=False, skipJER=False, tightMatch=False,
                 ak4tagger='deepJet'):

        self._output_location = 'outfiles-array/'

        self._year = year
        self._tagger  = tagger
        self._ak4tagger = ak4tagger
        self._jet_arbitration = jet_arbitration
        self._skipJER = skipJER
        self._tightMatch = tightMatch

        if self._ak4tagger == 'deepcsv':
            self._ak4tagBranch = 'btagDeepB'
        elif self._ak4tagger == 'deepJet':
            self._ak4tagBranch = 'btagDeepFlavB'
        else:
            raise NotImplementedError()

        self._btagSF = BTagCorrector('M', self._ak4tagger, year)

        with open('muon_triggers.json') as f:
            self._muontriggers = json.load(f)

        with open('triggers.json') as f:
            self._triggers = json.load(f)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2                                                                    
        with open('metfilters.json') as f:
            self._met_filters = json.load(f)

    def save_dfs_parquet(self, fname, dfs_dict):
        if self._output_location is not None:
            table = pa.Table.from_pandas(dfs_dict)
            if len(table) != 0:  # skip dataframes with empty entries                                                                           
                pq.write_table(table, self._output_location + "/parquet/" + fname + ".parquet")

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(output_collection[field])
        return output

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")

        if isRealData:
            # Nominal JEC are already applied in data
            return self.process_shift(events, None)

        if np.sum(ak.num(events.FatJet, axis=1)) < 1:
            return self.process_shift(events, None)

        jec_cache = {}

        thekey = f"{self._year}mc"
        if self._year == "2016":
            thekey = "2016postVFPmc"
        elif self._year == "2016APV":
            thekey = "2016preVFPmc"

        fatjets = fatjet_factory[thekey].build(add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache)
        jets = jet_factory[thekey].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        met = met_factory.build(events.MET, jets, {})

        shifts = [({"Jet": jets, "FatJet": fatjets, "MET": met}, None)]

        return processor.accumulate(self.process_shift(update(events, collections), name) for collections, name in shifts)

    def process_shift(self, events, shift_name):

        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        output = {dataset : processor.dict_accumulator()}
        weights = Weights(len(events), storeIndividual=True)

        if shift_name is None and not isRealData:
            output[dataset]['sumw'] = ak.sum(events.genWeight)

#        if len(events) == 0:
#            return output

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._triggers[self._year]:
                if t in events.HLT.fields:
                    trigger |= np.array(events.HLT[t])
            selection.add('trigger', trigger)
            del trigger
        else:
            selection.add('trigger', np.ones(len(events), dtype='bool'))

        if isRealData:
            selection.add('lumimask', lumiMasks[self._year[:4]](events.run, events.luminosityBlock))
        else:
            selection.add('lumimask', np.ones(len(events), dtype='bool'))

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._muontriggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
            selection.add('muontrigger', trigger)
            del trigger
        else:
            selection.add('muontrigger', np.ones(len(events), dtype='bool'))

        metfilter = np.ones(len(events), dtype='bool')
        for flag in self._met_filters[self._year]['data' if isRealData else 'mc']:
            metfilter &= np.array(events.Flag[flag])
        selection.add('metfilter', metfilter)
        del metfilter

        fatjets = events.FatJet
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['qcdrho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)

        candidatejet = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight  # this is loose in sampleContainer
        ]

        candidatejet = candidatejet[:, :2]  # Only consider first two to match generators
        if self._jet_arbitration == 'pt':
            candidatejet = ak.firsts(candidatejet)
        elif self._jet_arbitration == 'mass':
            candidatejet = ak.firsts(candidatejet[ak.argmax(candidatejet.msdcorr, axis=1, keepdims=True)])
        elif self._jet_arbitration == 'n2':
            candidatejet = ak.firsts(candidatejet[ak.argmin(candidatejet.n2ddt, axis=1, keepdims=True)])
        elif self._jet_arbitration == 'ddb':
            candidatejet = ak.firsts(candidatejet[ak.argmax(candidatejet.btagDDBvLV2, axis=1, keepdims=True)])
        elif self._jet_arbitration == 'ddc':
            candidatejet = ak.firsts(candidatejet[ak.argmax(candidatejet.btagDDCvLV2, axis=1, keepdims=True)])
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        if self._tagger == 'v1':
            bvl = candidatejet.btagDDBvL
            cvl = candidatejet.btagDDCvL
            cvb = candidatejet.btagDDCvB
        elif self._tagger == 'v2':
            bvl = candidatejet.btagDDBvLV2
            cvl = candidatejet.btagDDCvLV2
            cvb = candidatejet.btagDDCvBV2
        elif self._tagger == 'v3':
            bvl = candidatejet.particleNetMD_Xbb
            cvl = candidatejet.particleNetMD_Xcc / (1 - candidatejet.particleNetMD_Xbb)
            cvb = candidatejet.particleNetMD_Xcc / (candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
        elif self._tagger == 'v4':
            bvl = candidatejet.particleNetMD_Xbb
            cvl = candidatejet.btagDDCvLV2
            cvb = candidatejet.particleNetMD_Xcc / (candidatejet.particleNetMD_Xcc + candidatejet.particleNetMD_Xbb)
        else:
            raise ValueError("Not an option")

        selection.add('minjetkin',
            (candidatejet.pt >= 450)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr >= 40.)
            & (candidatejet.msdcorr < 201.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('minjetkinmu',
            (candidatejet.pt >= 400)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr >= 40.)
            & (candidatejet.msdcorr < 201.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jetid', candidatejet.isTight)
        selection.add('n2ddt', (candidatejet.n2ddt < 0.))
        if not self._tagger == 'v2':
            selection.add('ddbpass', (bvl >= 0.89))
        else:
            selection.add('ddbpass', (bvl >= 0.64))

        jets = events.Jet
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
            & (jets.puId > 0)
        ]
        # EE noise for 2017                                                                                                                           
        if self._year == '2017':
            jets = jets[
            (jets.pt > 50)
                | (abs(jets.eta) < 2.65)
                | (abs(jets.eta) > 3.139)
            ]

        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < self._btagSF._btagwp)
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagDeepB, axis=1, mask_identity=False) > self._btagSF._btagwp)

        met = events.MET
        selection.add('met', met.pt < 140.)

        # VBF specific variables
        dR = jets.delta_r(candidatejet)
        ak4_outside_ak8 = jets[dR > 0.8]

        jet1 = ak4_outside_ak8[:, 0:1]
        jet2 = ak4_outside_ak8[:, 1:2]

        deta = abs(ak.firsts(jet1).eta - ak.firsts(jet2).eta)
        dphi = (ak.firsts(jet1)).delta_phi(ak.firsts(jet2))
        mjj = ( ak.firsts(jet1) + ak.firsts(jet2) ).mass

        qgl1 = ak.firsts(jet1.qgl)
        qgl2 = ak.firsts(jet2.qgl)

        isvbf = ((deta > 3.5) & (mjj > 1000))
        isvbf = ak.fill_none(isvbf,False)
        selection.add('isvbf', isvbf)

        isnotvbf = ak.fill_none(~isvbf,True)
        selection.add('notvbf', isnotvbf)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
        )
        nmuons = ak.sum(goodmuon, axis=1)
        leadingmuon = ak.firsts(events.Muon[goodmuon])

        goodelectron = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nelectrons = ak.sum(goodelectron, axis=1)

        ntaus = ak.sum(
            (
                (events.Tau.pt > 20)
                & (abs(events.Tau.eta) < 2.3)
                & (events.Tau.rawIso < 5)
                & (events.Tau.idDeepTau2017v2p1VSjet)
                & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
                & ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2)
            ),
            axis=1,
        )

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons == 1) & (nelectrons == 0) & (ntaus == 0))
        selection.add('muonkin', (leadingmuon.pt > 55.) & (abs(leadingmuon.eta) < 2.1))
        selection.add('muonDphiAK8', abs(leadingmuon.delta_phi(candidatejet)) > 2*np.pi/3)

        if isRealData :
            genflavor = ak.zeros_like(candidatejet.pt)
        else:
            weights.add('genweight', events.genWeight)

            add_pileup_weight(weights, events.Pileup.nPU, self._year)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            if self._tightMatch:
                match_mask = ((candidatejet.pt - matchedBoson.pt)/matchedBoson.pt < 0.5) & ((candidatejet.msdcorr - matchedBoson.mass)/matchedBoson.mass < 0.3)
                selmatchedBoson = ak.mask(matchedBoson, match_mask)
                genflavor = bosonFlavor(selmatchedBoson)
            else:
                genflavor = bosonFlavor(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            add_VJets_kFactors(weights, events.GenPart, dataset)

            add_jetTriggerSF(weights, ak.firsts(fatjets), self._year, selection)

            add_muonSFs(weights, leadingmuon, self._year, selection)

            if self._year in ("2016APV", "2016", "2017"):
                weights.add("L1Prefiring", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)


            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd_matched = candidatejet.msdcorr * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)

        regions = {
            'signal-ggf': ['trigger','lumimask','metfilter','minjetkin','jetid','n2ddt','antiak4btagMediumOppHem','met','noleptons','notvbf'],
            'signal-vbf': ['trigger','lumimask','metfilter','minjetkin','jetid','n2ddt','antiak4btagMediumOppHem','met','noleptons','isvbf'],
            'muoncontrol': ['muontrigger','lumimask','metfilter','minjetkinmu', 'jetid', 'n2ddt', 'ak4btagMedium08', 'onemuon', 'muonkin', 'muonDphiAK8'],
            'noselection': [],
        }

        for k in ['genflavor','pt','n2ddt','msd','ddb','deta','dphi','mjj','weight']:
            output[dataset][k] = processor.column_accumulator(np.zeros(shape=(0,)))

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
            return ar

        vbfcuts = selection.all(*regions['signal-vbf'])
        output[dataset]['genflavor'] += processor.column_accumulator(normalize(genflavor,vbfcuts))
        output[dataset]['pt'] += processor.column_accumulator(normalize(candidatejet.pt,vbfcuts))
        output[dataset]['n2ddt'] += processor.column_accumulator(normalize(candidatejet.n2ddt,vbfcuts))
        output[dataset]['msd'] += processor.column_accumulator(normalize(msd_matched,vbfcuts))
        output[dataset]['ddb'] += processor.column_accumulator(normalize(bvl,vbfcuts))
        output[dataset]['deta'] += processor.column_accumulator(normalize(deta,vbfcuts))
        output[dataset]['dphi'] += processor.column_accumulator(normalize(dphi,vbfcuts))
        output[dataset]['mjj'] += processor.column_accumulator(normalize(mjj,vbfcuts))
        output[dataset]['weight'] += processor.column_accumulator(normalize(weights.weight(),vbfcuts))

        return output

    def postprocess(self, accumulator):
        return accumulator
