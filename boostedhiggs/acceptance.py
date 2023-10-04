import logging
import numpy as np
import awkward as ak
import json
import copy
from collections import defaultdict
from coffea import processor, hist
import hist as hist2
from coffea.analysis_tools import Weights, PackedSelection

from boostedhiggs.corrections import (
    add_HiggsEW_kFactors,

    # Jennet adds theory variations                                                                                                
    add_ps_weight,
    add_scalevar_7pt,
    add_scalevar_3pt,
    add_pdf_weight,
)

logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

class AccProcessor(processor.ProcessorABC):
    def __init__(self, systematics=True, ewkHcorr=True):
        self._ewkHcorr = ewkHcorr
        self._systematics = systematics

        self.make_output = lambda: {
            'sumw': processor.defaultdict_accumulator(float),
            'templates': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('mode', 'Mode', 3,0,3),
                hist.Bin('stxs', 'STXS bin', 28,0,28),
                hist.Bin('pth', 'Higgs $p_T$', [350, 400, 450]), 
            ),
        }

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        isQCDMC = 'QCD' in events.metadata['dataset']

        return self.process_shift(events, None)

    def process_shift(self, events, shift_name):

        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        isQCDMC = 'QCD' in dataset
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        output = self.make_output()
        if shift_name is None and not isRealData:
            output['sumw'][dataset] = ak.sum(events.genWeight)

        if len(events) == 0:
            return output

        if 'HToBB' in dataset:

            higgs = ak.firsts(events.GenPart[(events.GenPart.pdgId == 25) & events.GenPart.hasFlags(["fromHardProcess", "isLastCopy"])])

            stxs = events.HTXS.stage1_2_fine_cat_pTjet30GeV%100
            mode = events.HTXS.stage1_2_fine_cat_pTjet30GeV/100

            if self._ewkHcorr:
                    add_HiggsEW_kFactors(weights, events.GenPart, dataset)
                    
            if self._systematics:
                # Jennet adds theory variations                                                                               
                add_ps_weight(weights, events.PSWeight)
                if "LHEPdfWeight" in events.fields:
                    add_pdf_weight(weights,events.LHEPdfWeight)
                else:
                    add_pdf_weight(weights,[])
                if "LHEScaleWeight" in events.fields:
                    add_scalevar_7pt(weights, events.LHEScaleWeight)
                    add_scalevar_3pt(weights, events.LHEScaleWeight)
                else:
                    add_scalevar_7pt(weights,[])
                    add_scalevar_3pt(weights,[])

            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        def normalize(val, cut=None):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        import time
        tic = time.time()

        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]

            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)
                else:
                    weight = weights.weight()
            else:
                weight = weights.weight() * wmod

            output['templates'].fill(
                dataset=dataset,
                stxs=normalize(stxs),
                mode=normalize(mode),
                systematic=sname,
                pth=normalize(higgs.pt),
                weight=weight,
            )

        for systematic in systematics:
            if isRealData and systematic is not None:
                continue
            fill(systematic)

        toc = time.time()
        output["filltime"] = toc - tic
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
