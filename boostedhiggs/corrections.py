import numpy as np
import awkward as ak
import gzip
import pickle
import cloudpickle
import importlib.resources
import correctionlib
from coffea.lookup_tools.lookup_base import lookup_base
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea import lookup_tools
from coffea import util

with importlib.resources.path("boostedhiggs.data", "corrections.pkl.gz") as path:
    with gzip.open(path) as fin:
        compiled = pickle.load(fin)

ddt = util.load(f"boostedhiggs/data/ddtmap_n2b1_UL.coffea")
ddt_dict = {}

for year in ["2016APV", "2016", "2017","2018"]:
    h = ddt[year].to_hist()
    lookup = dense_lookup(h.view(), (h.axes[0].edges, h.axes[1].edges))
    ddt_dict[year] = lookup

# Updated for UL
with importlib.resources.path("boostedhiggs.data", "msdcorr.json") as filename:
    msdcorr = correctionlib.CorrectionSet.from_file(str(filename))

def corrected_msoftdrop(fatjets):
    msdraw = np.sqrt(
        np.maximum(
            0.0,
            (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum().mass2,
        )
    )
    msoftdrop = fatjets.msoftdrop
    msdfjcorr = msdraw / (1 - fatjets.rawFactor)

    corr = msdcorr["msdfjcorr"].evaluate(
        np.array(ak.flatten(msdfjcorr / fatjets.pt)),
        np.array(ak.flatten(np.log(fatjets.pt))),
        np.array(ak.flatten(fatjets.eta)),
    )
    corr = ak.unflatten(corr, ak.num(fatjets))
    corrected_mass = msdfjcorr * corr

    return corrected_mass

def n2ddt_shift(fatjets, year='2017'):
    return ddt_dict[year](fatjets.pt, fatjets.qcdrho)

def powheg_to_nnlops(genpt):
    return compiled['powheg_to_nnlops'](genpt)

# Jennet adds PDF + alpha_S weights
def add_pdf_weight(weights, pdf_weights):

    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)

    docstring = pdf_weights.__doc__

    # NNPDF31_nnlo_hessian_pdfas
    # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
    if True: #"306000 - 306102" in docstring:

        # Hessian PDF weights
        # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf                                   
        arg = pdf_weights[:,1:-2]-np.ones((nweights,100))
        summed = ak.sum(np.square(arg),axis=1)
        pdf_unc = np.sqrt( (1./99.) * summed )
        weights.add('PDF_weight', nom, pdf_unc + nom)

        # alpha_S weights
        # Eq. 27 of same ref
        as_unc = 0.5*(pdf_weights[:,102] - pdf_weights[:,101])
        weights.add('aS_weight', nom, as_unc + nom)

        # PDF + alpha_S weights
        # Eq. 28 of same ref
        pdfas_unc = np.sqrt( np.square(pdf_unc) + np.square(as_unc) )
        weights.add('PDFaS_weight', nom, pdfas_unc + nom) 

    else:
        weights.add('aS_weight', nom, up, down)
        weights.add('PDF_weight', nom, up, down)
        weights.add('PDFaS_weight', nom, up, down)

# Jennet adds 7 point scale variations
def add_scalevar_7pt(weights,var_weights):

    docstring = var_weights.__doc__

    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)
 
    if len(var_weights) > 0:
        if len(var_weights[0]) == 9: 
            up = np.maximum.reduce([var_weights[:,0],var_weights[:,1],var_weights[:,3],var_weights[:,5],var_weights[:,7],var_weights[:,8]])
            down = np.minimum.reduce([var_weights[:,0],var_weights[:,1],var_weights[:,3],var_weights[:,5],var_weights[:,7],var_weights[:,8]])
        elif len(var_weights[0]) > 1:
            print("Scale variation vector has length ", len(var_weights[0]))

    weights.add('scalevar_7pt', nom, up, down)

# Jennet adds 3 point scale variations
def add_scalevar_3pt(weights,var_weights):

    docstring = var_weights.__doc__

    nweights = len(weights.weight())

    nom   = np.ones(nweights)
    up    = np.ones(nweights)
    down  = np.ones(nweights)

    if len(var_weights) > 0:
        if len(var_weights[0]) == 9:
            up = np.maximum(var_weights[:,0], var_weights[:,8])
            down = np.minimum(var_weights[:,0], var_weights[:,8])
        elif len(var_weights[0]) > 1:
            print("Scale variation vector has length ", len(var_weights[0]))

    weights.add('scalevar_3pt', nom, up, down)

# Jennet adds PS weights
def add_ps_weight(weights,ps_weights):

    nweights = len(weights.weight())

    nom  = np.ones(nweights)

    up_isr   = np.ones(nweights)
    down_isr = np.ones(nweights)

    up_fsr   = np.ones(nweights)
    down_fsr = np.ones(nweights)

    if len(ps_weights[0]) == 4:
        up_isr = ps_weights[:,0]
        down_isr = ps_weights[:,2]

        up_fsr = ps_weights[:,1]
        down_fsr = ps_weights[:,3]
        
#        up = np.maximum.reduce([up_isr, up_fsr, down_isr, down_fsr])
#        down = np.minimum.reduce([up_isr, up_fsr, down_isr, down_fsr])

    elif len(ps_weights[0]) > 1:
        print("PS weight vector has length ", len(ps_weights[0]))

    weights.add('UEPS_ISR', nom, up_isr, down_isr)
    weights.add('UEPS_FSR', nom, up_fsr, down_fsr)


def add_pileup_weight(weights, nPU, year='2017'):
        weights.add(
            'pileup_weight',
            compiled[f'{year}_pileupweight'](nPU),
            compiled[f'{year}_pileupweight_puUp'](nPU),
            compiled[f'{year}_pileupweight_puDown'](nPU),
        )

with importlib.resources.path("boostedhiggs.data", "EWHiggsCorrections.json") as filename:
    hew_kfactors = correctionlib.CorrectionSet.from_file(str(filename))

def add_HiggsEW_kFactors(weights, genpart, dataset):
    """EW Higgs corrections"""
    def get_hpt():
        boson = ak.firsts(genpart[
            (genpart.pdgId == 25)
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ])
        return np.array(ak.fill_none(boson.pt, 0.))

    if "VBF" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["VBF_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("VBF_EW", ewknom)

    if "WplusH" in dataset or "WminusH" in dataset or "ZH" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["VH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("VH_EW", ewknom)

    if "ttH" in dataset:
        hpt = get_hpt()
        ewkcorr = hew_kfactors["ttH_EW"]
        ewknom = ewkcorr.evaluate(hpt)
        weights.add("ttH_EW", ewknom)

with importlib.resources.path("boostedhiggs.data", "ULvjets_corrections.json") as filename:
    vjets_kfactors = correctionlib.CorrectionSet.from_file(str(filename))

def add_VJets_kFactors(weights, genpart, dataset):
    """Revised version of add_VJets_NLOkFactor, for both NLO EW and ~NNLO QCD"""
    def get_vpt(check_offshell=False):
        """Only the leptonic samples have no resonance in the decay tree, and only
        when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
        """
        boson = ak.firsts(genpart[
            ((genpart.pdgId == 23)|(abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ])
        if check_offshell:
            offshell = genpart[
                genpart.hasFlags(["fromHardProcess", "isLastCopy"])
                & ak.is_none(boson)
                & (abs(genpart.pdgId) >= 11) & (abs(genpart.pdgId) <= 16)
            ].sum()
            return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
        return np.array(ak.fill_none(boson.pt, 0.))

    common_systs = [
        "d1K_NLO",
        "d2K_NLO",
        "d3K_NLO",
        "d1kappa_EW",
    ]
    zsysts = common_systs + [
        "Z_d2kappa_EW",
        "Z_d3kappa_EW",
    ]
    wsysts = common_systs + [
        "W_d2kappa_EW",
        "W_d3kappa_EW",
    ]

    def add_systs(systlist, qcdcorr, ewkcorr, vpt):
        ewknom = ewkcorr.evaluate("nominal", vpt)
        weights.add("vjets_nominal", qcdcorr * ewknom if qcdcorr is not None else ewknom)
        ones = np.ones_like(vpt)
        for syst in systlist:
            weights.add(syst, ones, ewkcorr.evaluate(syst + "_up", vpt) / ewknom, ewkcorr.evaluate(syst + "_down", vpt) / ewknom)

    if "ZJetsToQQ_HT" in dataset or "DYJetsToLL_M-50" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["ULZ_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["Z_FixedOrderComponent"]
        add_systs(zsysts, qcdcorr, ewkcorr, vpt)
    elif "WJetsToQQ_HT" in dataset or "WJetsToLNu" in dataset:
        vpt = get_vpt()
        qcdcorr = vjets_kfactors["ULW_MLMtoFXFX"].evaluate(vpt)
        ewkcorr = vjets_kfactors["W_FixedOrderComponent"]
        add_systs(wsysts, qcdcorr, ewkcorr, vpt)


with importlib.resources.path("boostedhiggs.data", "fatjet_triggerSF.json") as filename:
    jet_triggerSF = correctionlib.CorrectionSet.from_file(str(filename))


def add_jetTriggerSF(weights, leadingjet, year, selection):
    def mask(w):
        return np.where(selection.all('noleptons'), w, 1.)

    # Same for 2016 and 2016APV
    if '2016' in year:
        year = '2016'

    jet_pt = np.array(ak.fill_none(leadingjet.pt, 0.))
    jet_msd = np.array(ak.fill_none(leadingjet.msoftdrop, 0.))  # note: uncorrected
    nom = mask(jet_triggerSF[f'fatjet_triggerSF{year}'].evaluate("nominal", jet_pt, jet_msd))
    up = mask(jet_triggerSF[f'fatjet_triggerSF{year}'].evaluate("stat_up", jet_pt, jet_msd))
    down = mask(jet_triggerSF[f'fatjet_triggerSF{year}'].evaluate("stat_dn", jet_pt, jet_msd))
    weights.add('jet_trigger', nom, up, down)

with importlib.resources.path("boostedhiggs.data", "jec_compiled.pkl.gz") as path:
    with gzip.open(path) as fin:
        jmestuff = cloudpickle.load(fin)

jet_factory = jmestuff["jet_factory"]
fatjet_factory = jmestuff["fatjet_factory"]
met_factory = jmestuff["met_factory"]

def add_jec_variables(jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor)*jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor)*jets.mass
    jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets


def build_lumimask(filename):
    from coffea.lumi_tools import LumiMask
    with importlib.resources.path("boostedhiggs.data", filename) as path:
        return LumiMask(path)


lumiMasks = {
    "2016": build_lumimask("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"),
    "2017": build_lumimask("Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"),
    "2018": build_lumimask("Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"),
}

basedir = 'boostedhiggs/data/'
mutriglist = {
    '2016preVFP':{
        'TRIGNOISO':'NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt',
    },
    '2016postVFP':{
        'TRIGNOISO':'NUM_Mu50_or_TkMu50_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt',
    },
    '2017':{
        'TRIGNOISO':'NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt',
    },
    '2018':{
        'TRIGNOISO':'NUM_Mu50_or_OldMu100_or_TkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose_abseta_pt',
    },
}

ext = lookup_tools.extractor()
for year in ['2016preVFP','2016postVFP','2017','2018']:
    ext.add_weight_sets([f'muon_ID_{year}_value NUM_MediumPromptID_DEN_TrackerMuons_abseta_pt {basedir}Efficiencies_muon_generalTracks_Z_Run{year}_UL_ID.root'])
    ext.add_weight_sets([f'muon_ID_{year}_error NUM_MediumPromptID_DEN_TrackerMuons_abseta_pt_error {basedir}Efficiencies_muon_generalTracks_Z_Run{year}_UL_ID.root'])

    ext.add_weight_sets([f'muon_ISO_{year}_value NUM_LooseRelIso_DEN_MediumPromptID_abseta_pt {basedir}Efficiencies_muon_generalTracks_Z_Run{year}_UL_ISO.root'])
    ext.add_weight_sets([f'muon_ISO_{year}_error NUM_LooseRelIso_DEN_MediumPromptID_abseta_pt_error {basedir}Efficiencies_muon_generalTracks_Z_Run{year}_UL_ISO.root'])

    for trigopt in mutriglist[year]:
        trigname = mutriglist[year][trigopt]
        ext.add_weight_sets([f'muon_{trigopt}_{year}_value {trigname} {basedir}Efficiencies_muon_generalTracks_Z_Run{year}_UL_SingleMuonTriggers.root'])
        ext.add_weight_sets([f'muon_{trigopt}_{year}_error {trigname}_error {basedir}Efficiencies_muon_generalTracks_Z_Run{year}_UL_SingleMuonTriggers.root'])
ext.finalize()
lepsf_evaluator = ext.make_evaluator()
lepsf_keys = lepsf_evaluator.keys()

def add_muonSFs(weights, leadingmuon, year, selection):
    def mask(w):
        return np.where(selection.all('onemuon'), w, 1.)

    yeartag = year
    if year == '2016':
        yeartag = '2016postVFP'
    elif year == '2016APV':
        yeartag = '2016preVFP'

    for sf in lepsf_keys:

        if yeartag not in sf:
            continue
        if 'muon' not in sf:
            continue

        lep_pt = np.array(ak.fill_none(leadingmuon.pt, 0.))
        lep_eta = np.array(ak.fill_none(leadingmuon.eta, 0.))

        if 'value' in sf:
            nom = mask(lepsf_evaluator[sf](np.abs(lep_eta),lep_pt))
            shift = mask(lepsf_evaluator[sf.replace('_value','_error')](np.abs(lep_eta),lep_pt))

            weights.add(sf, nom, shift, shift=True)

