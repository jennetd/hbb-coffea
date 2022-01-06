import logging
import numpy as np
import awkward as ak
import json
import copy
from collections import defaultdict
from coffea import processor, hist
import hist as hist2
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from boostedhiggs.btag import BTagEfficiency, BTagCorrector
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
    add_VJets_NLOkFactor,
    add_VJets_kFactors,
    add_jetTriggerWeight,
    add_jetTriggerSF,
    add_mutriggerSF,
    add_mucorrectionsSF,
    jet_factory,
    fatjet_factory,
    add_jec_variables,
    met_factory,
    lumiMasks,

    # Jennet adds theory variations                                                                                                
#    add_ps_weight,
#    add_scalevar_7pt,
#    add_scalevar_3pt,
#    add_pdf_weight,
)


logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class VHbbProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='ddcvb', 
                 ):
        self._year = year
        self._tagger  = 'v2'
        self._ak4tagger = 'deepcsv'
        self._jet_arbitration = jet_arbitration
        self._skipJER = True
        self._tightMatch = False
        self._newVjetsKfactor= True
        self._newTrigger = True  # Fewer triggers, new maps (2017 only, ~no effect)

        if self._ak4tagger == 'deepcsv':
            self._ak4tagBranch = 'btagDeepB'
        elif self._ak4tagger == 'deepjet':
            self._ak4tagBranch = 'btagDeepFlavB'
        else:
            raise NotImplementedError()

        self._btagSF = BTagCorrector(year, self._ak4tagger, 'medium')

        self._msdSF = {
            '2016': 1.,
            '2017': 0.987,
            '2018': 0.970,
        }

        with open('muon_triggers.json') as f:
            self._muontriggers = json.load(f)

        with open('triggers.json') as f:
            self._triggers = json.load(f)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        self._met_filters = {
            '2016': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    # 'eeBadScFilter',
                ],
            },
            '2017': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
            },
            '2018': {
                'data': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
                'mc': [
                    'goodVertices',
                    'globalSuperTightHalo2016Filter',
                    'HBHENoiseFilter',
                    'HBHENoiseIsoFilter',
                    'EcalDeadCellTriggerPrimitiveFilter',
                    'BadPFMuonFilter',
                    'eeBadScFilter',
                    'ecalBadCalibFilterV2',
                ],
            },
        }

        self._json_paths = {
            '2016': 'jsons/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt',
            '2017': 'jsons/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt',
            '2018': 'jsons/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt',
        }

        optbins = np.r_[np.linspace(0, 0.15, 30, endpoint=False), np.linspace(0.15, 1, 86)]
        self.make_output = lambda: {
            'sumw': processor.defaultdict_accumulator(float),
            'btagWeight': hist2.Hist(
                hist2.axis.Regular(50, 0, 3, name='val', label='BTag correction'),
                hist2.storage.Weight(),
            ),
            'templates': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor1', 'Gen. jet 1 flavor', [1, 3, 4]),
                hist.Bin('genflavor2', 'Gen. jet 2 flavor', [1, 3, 4]),
                hist.Bin('msd1', r'Jet 1 $m_{sd}$', 23, 40, 201),
                hist.Bin('msd2', r'Jet 2 $m_{sd}$', 23, 40, 201),
                hist.Bin('ddb1', r'Jet 1 ddb score', 25,0,1),
                hist.Bin('ddc2', r'Jet 2 ddc score', [0,0.01,0.02,0.05,0.1,0.2,0.5,1]),
            ),
        }

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        isQCDMC = 'QCD' in events.metadata['dataset']

        if isRealData or isQCDMC:
            # Nominal JEC are already applied in data
            return self.process_shift(events, None)

        jec_cache = {}
        fatjets = fatjet_factory[f"{self._year}mc"].build(add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache)
        jets = jet_factory[f"{self._year}mc"].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        met = met_factory.build(events.MET, jets, {})

        shifts = [
            ({"Jet": jets, "FatJet": fatjets, "MET": met}, None),
        ]
        return processor.accumulate(self.process_shift(update(events, collections), name) for collections, name in shifts)

    def process_shift(self, events, shift_name):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        isQCDMC = 'QCD' in dataset
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        output = self.make_output()
        if shift_name is None and not isRealData:
            output['sumw'][dataset] = ak.sum(events.genWeight)

        if isRealData or self._newTrigger:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._triggers[self._year]:
                if t in events.HLT.fields:
                    trigger |= np.array(events.HLT[t])
            selection.add('trigger', trigger)
            del trigger
        else:
            selection.add('trigger', np.ones(len(events), dtype='bool'))

        if isRealData:
            selection.add('lumimask', lumiMasks[self._year](events.run, events.luminosityBlock))
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
        fatjets['msdcorr_full'] = fatjets['msdcorr'] * self._msdSF[self._year]

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
        elif self._jet_arbitration == 'ddcvb':
            leadingjets = candidatejet[:, 0:2]
            # ascending = true                                                                                                                                
            indices = ak.argsort(leadingjets.btagDDCvBV2,axis=1)

            # candidate jet is more b-like                                                                                                               
            candidatejet = ak.firsts(leadingjets[indices[:, 0:1]])
            # second jet is more charm-like                                                                                                              
            secondjet = ak.firsts(leadingjets[indices[:, 1:2]])
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        bvl1 = candidatejet.btagDDBvLV2
        cvl1 = candidatejet.btagDDCvLV2
        cvb1 = candidatejet.btagDDCvBV2

        bvl2 = secondjet.btagDDBvLV2
        cvl2 = secondjet.btagDDCvLV2
        cvb2 = secondjet.btagDDCvBV2

        selection.add('jet1kin',
            (candidatejet.pt >= 450)
            & (candidatejet.msdcorr >= 47.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jet2kin',
            (secondjet.pt >= 400)
            & (secondjet.msdcorr >= 47.)
            & (abs(secondjet.eta) < 2.5)
        )

        selection.add('jetacceptance',
            (candidatejet.msdcorr >= 40.)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr < 201.)
            & (secondjet.msdcorr >= 40.)
            & (secondjet.pt < 1200)
            & (secondjet.msdcorr < 201.)
        )

        selection.add('jetid',
                      candidatejet.isTight
                      & secondjet.isTight
        )
        selection.add('n2ddt',
                      (candidatejet.n2ddt < 0.)
                      & (secondjet.n2ddt < 0.)
        )

        selection.add('ddbpass', (bvl1 >= 0.64))

        jets = events.Jet
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
        ]
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._ak4tagger][self._year]['medium'])
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._ak4tagger][self._year]['medium'])

        met = events.MET
        selection.add('met', met.pt < 140.)

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
                & events.Tau.idDecayMode
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
            genflavor1 = ak.zeros_like(candidatejet.pt)
            genflavor2 = ak.zeros_like(secondjet.pt)
        else:
            weights.add('genweight', events.genWeight)

            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson1 = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            matchedBoson2 = secondjet.nearest(bosons, axis=None, threshold=0.8)

            genflavor1 = bosonFlavor(matchedBoson1)
            genflavor2 = bosonFlavor(matchedBoson2)

            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            if self._newVjetsKfactor:
                add_VJets_kFactors(weights, events.GenPart, dataset)
            else:
                add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)

            if shift_name is None:
                output['btagWeight'].fill(val=self._btagSF.addBtagWeight(weights, ak4_away, self._ak4tagBranch))

            if self._newTrigger:
                add_jetTriggerSF(weights, ak.firsts(fatjets), self._year, selection)
            else:
                add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)

            add_mutriggerSF(weights, leadingmuon, self._year, selection)
            add_mucorrectionsSF(weights, leadingmuon, self._year, selection)

            if self._year in ("2016", "2017"):
                weights.add("L1Prefiring", events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn)


            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd1_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor1 > 0) + candidatejet.msdcorr * (genflavor1 == 0)
        msd2_matched = secondjet.msdcorr * self._msdSF[self._year] * (genflavor2 > 0) + secondjet.msdcorr * (genflavor2 == 0)

        regions = {
            'signal': ['trigger','lumimask','metfilter','jet1kin','jet2kin','jetid','jetacceptance','n2ddt','met','noleptons'],
        }

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        import time
        tic = time.time()

        systematics = [None] #+ list(weights.variations)

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]

            output['templates'].fill(
                dataset=dataset,
                region=region,
                genflavor1=normalize(genflavor1,cut),
                genflavor2=normalize(genflavor2,cut),
                msd1=normalize(msd1_matched, cut),
                msd2=normalize(msd2_matched, cut),
                ddb1=normalize(bvl1, cut),
                ddc2=normalize(cvl2, cut),
                weight=weight,
            )

        for region in regions:
            for systematic in systematics:
                if isRealData and systematic is not None:
                    continue
                fill(region, systematic)

        toc = time.time()
        output["filltime"] = toc - tic
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
