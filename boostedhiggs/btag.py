import os
import numpy
import logging
import awkward
from coffea import processor, hist, util
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.btag_tools import BTagScaleFactor


logger = logging.getLogger(__name__)


class BTagEfficiency(processor.ProcessorABC):
    btagWPs = {
        'deepcsv': {
            '2016': {
                'medium': 0.6321,
            },
            '2017': {
                'medium': 0.4941,
            },
            '2018': {
                'medium': 0.4184,
            },
        },
        'deepjet': {
            '2016': {
                'medium': 0.3093,
            },
            '2017': {
                'medium': 0.3033,
            },
            '2018': {
                'medium': 0.2770,
            },
        },
    }

    def __init__(self, year='2017'):
        self._year = year
        self._accumulator = hist.Hist(
            'Events',
            hist.Cat('dataset', 'Dataset'),
            hist.Cat('tagger', 'Tagger'),
            hist.Cat('btag', 'BTag WP pass/fail'),
            hist.Bin('flavor', 'Jet hadronFlavour', [0, 4, 5, 6]),
            hist.Bin('pt', 'Jet pT', [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]),
            hist.Bin('abseta', 'Jet abseta', [0, 1.4, 2.0, 2.5]),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & (events.Jet.jetId & 2)  # tight id
        ]

        out = self.accumulator.identity()
        tags = [
            ('deepcsv', 'btagDeepB', 'medium'),
            ('deepjet', 'btagDeepFlavB', 'medium'),
        ]

        for tagger, branch, wp in tags:
            passbtag = jets[branch] > BTagEfficiency.btagWPs[tagger][self._year][wp]

            out.fill(
                dataset=events.metadata['dataset'],
                tagger=tagger,
                # tagger=f"{tagger}_{wp}",
                btag='pass',
                flavor=awkward.flatten(jets[passbtag].hadronFlavour),
                pt=awkward.flatten(jets[passbtag].pt),
                abseta=awkward.flatten(abs(jets[passbtag].eta)),
            )
            out.fill(
                dataset=events.metadata['dataset'],
                tagger=tagger,
                # tagger=f"{tagger}_{wp}",
                btag='fail',
                flavor=awkward.flatten(jets[~passbtag].hadronFlavour),
                pt=awkward.flatten(jets[~passbtag].pt),
                abseta=awkward.flatten(abs(jets[~passbtag].eta)),
            )

        return out

    def postprocess(self, a):
        return a


class BTagCorrector:
    def __init__(self, year, tagger, workingpoint):
        self._year = year
        self._wp = BTagEfficiency.btagWPs[tagger][year][workingpoint]
        files = {
            "deepcsv" : {
                '2016': 'DeepCSV_Moriond17_B_H.csv.gz',
                '2017': 'DeepCSV_94XSF_V5_B_F.csv.gz',
                '2018': 'DeepCSV_102XSF_V1.csv.gz',
            },
            "deepjet" : {
                '2016': 'DeepJet_2016LegacySF_WP_V1.csv.gz',
                '2017': 'DeepFlavour_94XSF_WP_V3_B_F.csv.gz',
                '2018': 'DeepJet_102XSF_WP_V1.csv.gz',
            },
        }
        filename = os.path.join(os.path.dirname(__file__), 'data', files[tagger][year])
        self.sf = BTagScaleFactor(filename, workingpoint)
        # files = {
        #     '2016': 'btagQCD2016.coffea',
        #     '2017': 'btagQCD2017.coffea',
        #     '2018': 'btagQCD2018.coffea',
        # }
        files = {
            '2016': 'btagEff2016.coffea',
            '2017': 'btagEff2017.coffea',
            '2018': 'btagEff2018.coffea',
        }
        filename = os.path.join(os.path.dirname(__file__), 'data', files[year])
        btag = util.load(filename).integrate('tagger', tagger).integrate('dataset')
        # bpass = btag.integrate('btag', 'pass').values()[()]
        # ball = btag.integrate('btag').values()[()]
        bpass = btag.integrate('btag', 'pass').values()[()]
        ball = btag.integrate('btag').values()[()]
        nom = bpass / numpy.maximum(ball, 1.)
        dn, up = hist.clopper_pearson_interval(bpass, ball)
        self.eff = dense_lookup(nom, [ax.edges() for ax in btag.axes()[1:]])
        self.eff_statUp = dense_lookup(up, [ax.edges() for ax in btag.axes()[1:]])
        self.eff_statDn = dense_lookup(dn, [ax.edges() for ax in btag.axes()[1:]])

    def addBtagWeight(self, weights, jets, tagger_branch):
        abseta = abs(jets.eta)
        passbtag = jets[tagger_branch] > self._wp

        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1a_Event_reweighting_using_scale
        def combine(eff, sf):
            # tagged SF = SF*eff / eff = SF
            tagged_sf = awkward.prod(sf[passbtag], axis=-1)
            # untagged SF = (1 - SF*eff) / (1 - eff)
            untagged_sf = awkward.prod(((1 - sf*eff) / (1 - eff))[~passbtag], axis=-1)
            return awkward.fill_none(tagged_sf * untagged_sf, 1.)  # TODO: move None guard to coffea

        eff_nom = self.eff(jets.hadronFlavour, jets.pt, abseta)
        eff_statUp = self.eff_statUp(jets.hadronFlavour, jets.pt, abseta)
        eff_statDn = self.eff_statDn(jets.hadronFlavour, jets.pt, abseta)
        sf_nom = self.sf.eval('central', jets.hadronFlavour, abseta, jets.pt)
        sf_systUp = self.sf.eval('up', jets.hadronFlavour, abseta, jets.pt)
        sf_systDn = self.sf.eval('down', jets.hadronFlavour, abseta, jets.pt)

        nom = combine(eff_nom, sf_nom)
        weights.add('btagWeight', nom, weightUp=combine(eff_nom, sf_systUp), weightDown=combine(eff_nom, sf_systDn))
        weights.add('btagEffStat', numpy.ones(len(nom)), weightUp=combine(eff_statUp, sf_nom) / nom, weightDown=combine(eff_statDn, sf_nom) / nom)
        for i in numpy.where((nom < 0.01) | (nom > 10) | numpy.isnan(nom))[0][:4]:
            jet = jets[i]
            logger.info("Strange weight for event: %r", nom[i])
            logger.info("    jet pts: %r", jet.pt)
            logger.info("    jet etas: %r", jet.eta)
            logger.info("    jet flavors: %r", jet.hadronFlavour)
            logger.info("    jet btags: %r", jet.btagDeepB)
            logger.info("    result eff: %r up %r down %r", eff_nom[i], eff_statUp[i], eff_statDn[i])
            logger.info("    result sf: %r", sf_nom[i])
        return nom


if __name__ == '__main__':
    tagger = 'deepcsv'
    b = BTagCorrector('2017', tagger, 'medium')
    b.sf.eval('central', numpy.array([0, 1, 2]), numpy.array([-2.3, 2., 0.]), numpy.array([20.1, 300., 10.]))
    b.sf.eval('down_uncorrelated', numpy.array([2, 2, 2]), numpy.array([-2.6, 2.9, 0.]), numpy.array([20.1, 300., 1000.]))
    import awkward as ak
    b.sf.eval('central', ak.Array([[0], [1, 2]]), ak.Array([[-2.3], [2., 0.]]), ak.Array([[20.1], [300., 10.]]))
    import pickle
    bb = pickle.loads(pickle.dumps(b))
    bb.sf.eval('central', ak.Array([[0], [1, 2]]), ak.Array([[-2.3], [2., 0.]]), ak.Array([[20.1], [300., 10.]]))
    b2 = BTagCorrector('2016', tagger, 'medium')
    b3 = BTagCorrector('2018', tagger, 'medium')
