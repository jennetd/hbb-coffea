from __future__ import print_function, division
import sys, os
import csv, json
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats
import pickle
import ROOT
import pandas as pd

import rhalphalib as rl
from rhalphalib import AffineMorphTemplate, MorphHistW2

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False

eps=0.001
do_systematics = True
do_muon_CR = True

def badtemp_ma(hvalues, mask=None):
    # Need minimum size & more than 1 non-zero bins           
    tot = np.sum(hvalues[mask])
    
    count_nonzeros = np.sum(hvalues[mask] > 0)
    if (tot < eps) or (count_nonzeros < 2):
        return True
    else:
        return False

def syst_variation(numerator,denominator):
    """
    Get systematic variation relative to nominal (denominator)
    """
    var = np.divide(numerator,denominator)
    var[np.where(numerator==0)] = 1
    var[np.where(denominator==0)] = 1

    return var

def smass(sName):
    if sName in ['ggF','VBF','WH','ZH','ttH']:
        _mass = 125.
    elif sName in ['Wjets','EWKW','ttbar','singlet','VV']:
        _mass = 80.379
    elif sName in ['Zjets','Zjetsbb','EWKZ']:
        _mass = 91.
    else:
        raise ValueError("What is {}".format(sName))
    return _mass

def one_bin(sName, passed, ptbin, cat, obs, syst, muon=False):
    f = ROOT.TFile.Open('signalregion.root')
    if muon:
        f = ROOT.TFile.Open('muonCR.root')

    name = cat+'fail_'
    if passed:
        name = cat+'pass_'
    if cat == 'ggf_':
        name += 'pt'+str(ptbin)+'_'
    if cat == 'vbf_':
        name += 'mjj'+str(ptbin)+'_'

    name += sName+'_'+syst

    h = f.Get(name)
    newh = h.Rebin(h.GetNbinsX())
    sumw = [newh.GetBinContent(1)]
    sumw2 = [newh.GetBinError(1)]

    return (np.array(sumw), np.array([0., 1.]), "onebin", np.array(sumw2))

def get_template(sName, passed, ptbin, cat, obs, syst, muon=False):
    """
    Read msd template from root file
    """

    f = ROOT.TFile.Open('signalregion.root')
    if muon:
        f = ROOT.TFile.Open('muonCR.root')

    name = cat+'fail_'
    if passed:
        name = cat+'pass_'
    if cat == 'ggf_':
        name += 'pt'+str(ptbin)+'_'
    if cat == 'vbf_':
        name += 'mjj'+str(ptbin)+'_'

    name += sName+'_'+syst

    h = f.Get(name)

    sumw = []
    sumw2 = []

    for i in range(1,h.GetNbinsX()+1):

        if h.GetBinContent(i) < 0:
#            print('negative bin',name)
            sumw += [0]
            sumw2 += [0]
        else:
            sumw += [h.GetBinContent(i)]
            sumw2 += [h.GetBinError(i)*h.GetBinError(i)]

    return (np.array(sumw), obs.binning, obs.name, np.array(sumw2))

def shape_to_num(var, nom, clip=1.5):
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)

    if abs(var_rate/nom_rate) > clip:
        var_rate = clip*nom_rate

    if var_rate < 0:
        var_rate = 0

    return var_rate/nom_rate

def passfailSF(isPass, sName, ptbin, cat, obs, mask, SF=1, SF_unc_up=0.1, SF_unc_down=-0.1, muon=False):
    """
    Return (SF, SF_unc) for a pass/fail scale factor.
    """
    if isPass:
        return SF, 1. + SF_unc_up / SF, 1. + SF_unc_down / SF
    else:
        _pass = get_template(sName, 1, ptbin+1, cat, obs=obs, syst='nominal', muon=muon)
        _pass_rate = np.sum(_pass[0]*mask)

        _fail = get_template(sName, 0, ptbin+1, cat, obs=obs, syst='nominal', muon=muon)
        _fail_rate = np.sum(_fail[0]*mask)

        if _fail_rate > 0:
            _sf = 1 + (1 - SF) * _pass_rate / _fail_rate
            _sfunc_up = 1. - SF_unc_up * (_pass_rate / _fail_rate)
            _sfunc_down = 1. - SF_unc_down * (_pass_rate / _fail_rate)

            return _sf, _sfunc_up, _sfunc_down
        else:
            return 1, 1, 1

def plot_mctf(tf_MCtempl, msdbins, name):
    """
    Plot the MC pass / fail TF as function of (pt,rho) and (pt,msd)
    """
    import matplotlib.pyplot as plt

    # arrays for plotting pt vs msd                    
    pts = np.linspace(450,1200,15)
    ptpts, msdpts = np.meshgrid(pts[:-1] + 0.5 * np.diff(pts), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
    ptpts_scaled = (ptpts - 450.) / (1200. - 450.)
    rhopts = 2*np.log(msdpts/ptpts)

    rhopts_scaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins]
    msdpts = msdpts[validbins]
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)
    df = pd.DataFrame([])
    df['msd'] = msdpts.reshape(-1)
    df['pt'] = ptpts.reshape(-1)
    df['MCTF'] = tf_MCtempl_vals.reshape(-1)

    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["msd"],y=df["pt"],weights=df["MCTF"], bins=(msdbins,pts))
    plt.xlabel("$m_{sd}$ [GeV]")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3],ax=ax)
    cb.set_label("Ratio")
    fig.savefig("plots/MCTF_msdpt_"+name+".png",bbox="tight")
    fig.savefig("plots/MCTF_msdpt_"+name+".pdf",bbox="tight")
    plt.clf()

    # arrays for plotting pt vs rho                                          
    rhos = np.linspace(-6,-2.1,23)
    ptpts, rhopts = np.meshgrid(pts[:-1] + 0.5*np.diff(pts), rhos[:-1] + 0.5 * np.diff(rhos), indexing='ij')
    ptpts_scaled = (ptpts - 450.) / (1200. - 450.)
    rhopts_scaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins]
    rhopts = rhopts[validbins]
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)

    df = pd.DataFrame([])
    df['rho'] = rhopts.reshape(-1)
    df['pt'] = ptpts.reshape(-1)
    df['MCTF'] = tf_MCtempl_vals.reshape(-1)

    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["rho"],y=df["pt"],weights=df["MCTF"],bins=(rhos,pts))
    plt.xlabel("rho")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3],ax=ax)
    cb.set_label("Ratio")
    fig.savefig("plots/MCTF_rhopt_"+name+".png",bbox="tight")
    fig.savefig("plots/MCTF_rhopt_"+name+".pdf",bbox="tight")

    return

def ggfvbf_rhalphabet(tmpdir,
                    throwPoisson = True,
                    fast=0):
    """ 
    Create the data cards!
    """
    with open('sf.json') as f:
        SF = json.load(f)

    with open('lumi.json') as f:
        lumi = json.load(f)

    # TT params
    tqqeffSF = rl.IndependentParameter('tqqeffSF_{}'.format(year), 1., -50, 50)
    tqqnormSF = rl.IndependentParameter('tqqnormSF_{}'.format(year), 1., -50, 50)

    # Systematics
    sys_lumi_uncor = rl.NuisanceParameter('CMS_lumi_13TeV_{}'.format(year[:4]), 'lnN')
    sys_lumi_cor_161718 = rl.NuisanceParameter('CMS_lumi_13TeV_correlated', 'lnN')
    sys_lumi_cor_1718 = rl.NuisanceParameter('CMS_lumi_13TeV_correlated_20172018', 'lnN')

    sys_eleveto = rl.NuisanceParameter('CMS_hbb_e_veto_{}'.format(year), 'lnN')                                    
    sys_muveto = rl.NuisanceParameter('CMS_hbb_mu_veto_{}'.format(year), 'lnN')  
    sys_tauveto = rl.NuisanceParameter('CMS_hbb_tau_veto_{}'.format(year), 'lnN')

    sys_dict = {}

    # Systematics 

    # experimental systematics are mostly uncorrelated across years
    yearstr = year
    if '2016' in year:
        if 'APV' in year:
            yearstr = '2016preVFP'
        else:
            yearstr = '2016postVFP'

    sys_dict['muon_ID_'+yearstr+'_value'] = rl.NuisanceParameter('CMS_mu_id_{}'.format(year), 'lnN')
    sys_dict['muon_ISO_'+yearstr+'_value'] = rl.NuisanceParameter('CMS_mu_iso_{}'.format(year), 'lnN')
    sys_dict['muon_TRIGNOISO_'+yearstr+'_value'] = rl.NuisanceParameter('CMS_hbb_mu_trigger_{}'.format(year), 'lnN')

    sys_dict['JES'] = rl.NuisanceParameter('CMS_scale_j_{}'.format(year), 'lnN')
    sys_dict['JER'] = rl.NuisanceParameter('CMS_res_j_{}'.format(year), 'lnN')
    sys_dict['UES'] = rl.NuisanceParameter('CMS_ues_j_{}'.format(year), 'lnN')
    sys_dict['jet_trigger'] = rl.NuisanceParameter('CMS_hbb_jet_trigger_{}'.format(year), 'lnN')
    sys_dict['pileup_weight'] = rl.NuisanceParameter('CMS_hbb_PU_{}'.format(year), 'lnN')

    sys_dict['btagSFlight_'+year] = rl.NuisanceParameter('CMS_hbb_btagSFlight_{}'.format(year), 'lnN')
    sys_dict['btagSFbc_'+year] = rl.NuisanceParameter('CMS_hbb_btagSFbc_{}'.format(year), 'lnN')
    sys_dict['btagSFlight_correlated'] = rl.NuisanceParameter('CMS_hbb_btagSFlight_correlated'.format(year), 'lnN')
    sys_dict['btagSFbc_correlated'] = rl.NuisanceParameter('CMS_hbb_btagSFbc_correlated', 'lnN')

    sys_dict['L1Prefiring'] = rl.NuisanceParameter('CMS_L1Prefiring_{}'.format(year),'lnN')

    exp_systs = ['pileup_weight',
                 'JES','JER','UES',
                 'btagSFlight_'+year,'btagSFbc_'+year,
                 'btagSFlight_correlated','btagSFbc_correlated']

    if '2018' not in year:
        exp_systs += ['L1Prefiring']
    mu_exp_systs = exp_systs + ['muon_ID_'+yearstr+'_value','muon_ISO_'+yearstr+'_value','muon_TRIGNOISO_'+yearstr+'_value']

    exp_systs += ['jet_trigger']

    sys_ddxeffbb = rl.NuisanceParameter('CMS_eff_bb_{}'.format(year), 'lnN')
    sys_ddb_pt_1 = rl.NuisanceParameter('CMS_hbb_ddb_1_{}'.format(year), 'lnN')
    sys_ddb_pt_2 = rl.NuisanceParameter('CMS_hbb_ddb_2_{}'.format(year), 'lnN')
    sys_ddb_pt_3 = rl.NuisanceParameter('CMS_hbb_ddb_3_{}'.format(year), 'lnN')
    sys_ddb_pt = [sys_ddb_pt_1,sys_ddb_pt_1,sys_ddb_pt_1,sys_ddb_pt_2,sys_ddb_pt_2,sys_ddb_pt_3]

    sys_veff = rl.NuisanceParameter('CMS_hbb_veff_{}'.format(year), 'lnN')
    sys_veff_pt_1 = rl.NuisanceParameter('CMS_hbb_veff_1_{}'.format(year), 'lnN')
    sys_veff_pt_2 = rl.NuisanceParameter('CMS_hbb_veff_2_{}'.format(year), 'lnN')
    sys_veff_pt_3 = rl.NuisanceParameter('CMS_hbb_veff_3_{}'.format(year), 'lnN')
    sys_veff_pt = [sys_veff_pt_1,sys_veff_pt_1,sys_veff_pt_1,sys_veff_pt_2,sys_veff_pt_2,sys_veff_pt_3]

    sys_smear = rl.NuisanceParameter('CMS_hbb_smear_{}'.format(year), 'shape')

    sys_scale = rl.NuisanceParameter('CMS_hbb_scale_{}'.format(year), 'shape')
    sys_scale_pt_1 = rl.NuisanceParameter('CMS_hbb_scale_pt_1_{}'.format(year), 'shape')
    sys_scale_pt_2 = rl.NuisanceParameter('CMS_hbb_scale_pt_2_{}'.format(year), 'shape')
    sys_scale_pt_3 = rl.NuisanceParameter('CMS_hbb_scale_pt_3_{}'.format(year), 'shape')
    sys_scale_pt_4 = rl.NuisanceParameter('CMS_hbb_scale_pt_4_{}'.format(year), 'shape')
    sys_scale_pt_5 = rl.NuisanceParameter('CMS_hbb_scale_pt_5_{}'.format(year), 'shape')
    sys_scale_pt_6 = rl.NuisanceParameter('CMS_hbb_scale_pt_6_{}'.format(year), 'shape')
    sys_scale_pt = [sys_scale_pt_1,sys_scale_pt_2,sys_scale_pt_3,sys_scale_pt_4,sys_scale_pt_5,sys_scale_pt_6]

    sys_vbf_ttbar_unc = rl.NuisanceParameter('CMS_hbb_vbfmucr_{}'.format(year), 'lnN')

    # theory systematics are correlated across years
    for sys in ['d1kappa_EW', 'Z_d2kappa_EW', 'Z_d3kappa_EW', 'W_d2kappa_EW', 'W_d3kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO']:
        sys_dict[sys] = rl.NuisanceParameter('CMS_hbb_{}'.format(sys), 'lnN')
            
    Zjets_thsysts = ['d1kappa_EW', 'Z_d2kappa_EW', 'Z_d3kappa_EW', 'd1K_NLO', 'd2K_NLO']
    Wjets_thsysts = ['d1kappa_EW', 'W_d2kappa_EW', 'W_d3kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO']         
                      
    pdf_Higgs_ggF = rl.NuisanceParameter('pdf_Higgs_ggF','lnN')
    pdf_Higgs_VBF = rl.NuisanceParameter('pdf_Higgs_VBF','lnN')
    pdf_Higgs_VH  = rl.NuisanceParameter('pdf_Higgs_VH','lnN')
    pdf_Higgs_ttH = rl.NuisanceParameter('pdf_Higgs_ttH','lnN')

    scale_ggF = rl.NuisanceParameter('QCDscale_ggF', 'lnN')
    scale_VBF = rl.NuisanceParameter('QCDscale_VBF', 'lnN')
    scale_VH = rl.NuisanceParameter('QCDscale_VH', 'lnN')
    scale_ttH = rl.NuisanceParameter('QCDscale_ttH', 'lnN')

    isr_ggF = rl.NuisanceParameter('UEPS_ISR_ggF', 'lnN')
    isr_VBF = rl.NuisanceParameter('UEPS_ISR_VBF', 'lnN')
    isr_VH = rl.NuisanceParameter('UEPS_ISR_VH', 'lnN')
    isr_ttH = rl.NuisanceParameter('UEPS_ISR_ttH', 'lnN')

    fsr_ggF = rl.NuisanceParameter('UEPS_FSR_ggF', 'lnN')
    fsr_VBF = rl.NuisanceParameter('UEPS_FSR_VBF', 'lnN')
    fsr_VH = rl.NuisanceParameter('UEPS_FSR_VH', 'lnN')
    fsr_ttH = rl.NuisanceParameter('UEPS_FSR_ttH', 'lnN')

    # define bins    
    ptbins = {}
    ptbins['ggf'] = np.array([450, 500, 550, 600, 675, 800, 1200])
    ptbins['vbf'] = np.array([450,1200])

    mjjbins = {}
    mjjbins['ggf'] = np.array([0,13000])
    mjjbins['vbf'] = np.array([1000,2000,13000])

    npt = {}
    npt['ggf'] = len(ptbins['ggf']) - 1
    npt['vbf'] = len(ptbins['vbf']) - 1

    nmjj = {}
    nmjj['ggf'] = len(mjjbins['ggf']) - 1
    nmjj['vbf'] = len(mjjbins['vbf']) - 1

    msdbins = np.linspace(40, 201, 24)
    msd = rl.Observable('msd', msdbins)

    validbins = {}

    cats = ['vbf']
    ncat = len(cats)

    Nfail_qcd_MC = 0
    Nfail_data = 0

    # Build qcd MC pass+fail model and fit to polynomial
    tf_params = {}
    for cat in cats:

        fitfailed_qcd = 0

        # here we derive these all at once with 2D array                            
        ptpts, msdpts = np.meshgrid(ptbins[cat][:-1] + 0.3 * np.diff(ptbins[cat]), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
        rhopts = 2*np.log(msdpts/ptpts)
        ptscaled = (ptpts - 450.) / (1200. - 450.)
        rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
        validbins[cat] = (rhoscaled >= 0) & (rhoscaled <= 1)
        rhoscaled[~validbins[cat]] = 1  # we will mask these out later   

        while fitfailed_qcd < 5:
        
            qcdmodel = rl.Model('qcdmodel_'+cat)
            qcdpass, qcdfail = 0., 0.

            for ptbin in range(npt[cat]):
                for mjjbin in range(nmjj[cat]):

                    failCh = rl.Channel('ptbin%dmjjbin%d%s%s%s' % (ptbin, mjjbin, cat, 'fail',year))
                    passCh = rl.Channel('ptbin%dmjjbin%d%s%s%s' % (ptbin, mjjbin, cat, 'pass',year))
                    qcdmodel.addChannel(failCh)
                    qcdmodel.addChannel(passCh)

                    binindex = ptbin
                    if cat == 'vbf':
                        binindex = mjjbin

                    # QCD templates from file                           
                    failTempl = get_template('QCD', 0, binindex+1, cat+'_', obs=msd, syst='nominal')
                    passTempl = get_template('QCD', 1, binindex+1, cat+'_', obs=msd, syst='nominal')

                    failCh.setObservation(failTempl, read_sumw2=True)
                    passCh.setObservation(passTempl, read_sumw2=True)

                    qcdfail += sum([val for val in failCh.getObservation()[0]])
                    qcdpass += sum([val for val in passCh.getObservation()[0]])

            qcdeff = qcdpass / qcdfail
            print('Inclusive P/F from Monte Carlo = ' + str(qcdeff))

            Nfail_qcd_MC += qcdfail

            # initial values                                                                 
            print('Initial fit values read from file initial_vals*')
            with open('initial_vals_'+cat+'.json') as f:
                initial_vals = np.array(json.load(f)['initial_vals'])
            print(initial_vals)

            print("TFpf order " + str(initial_vals.shape[0]-1) + " in pT, " + str(initial_vals.shape[1]-1) + " in rho")
            tf_MCtempl = rl.BasisPoly("tf_MCtempl_"+cat+year,
                                      (initial_vals.shape[0]-1,initial_vals.shape[1]-1),
                                      ['pt', 'rho'], 
                                      basis='Bernstein',
                                      init_params=initial_vals,
                                      limits=(-50, 50), coefficient_transform=None)

            tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)

            for ptbin in range(npt[cat]):
                for mjjbin in range(nmjj[cat]):

                    failCh = qcdmodel['ptbin%dmjjbin%d%sfail%s' % (ptbin, mjjbin, cat, year)]
                    passCh = qcdmodel['ptbin%dmjjbin%d%spass%s' % (ptbin, mjjbin, cat, year)]
                    failObs = failCh.getObservation()
                    passObs = passCh.getObservation()
                
                    qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%dmjjbin%d%s%s_%d' % (ptbin, mjjbin, cat, year, i), 0, -50, 50) for i in range(msd.nbins)])
                    sigmascale = 10.
                    scaledparams = failObs * (1 + sigmascale/np.maximum(1., np.sqrt(failObs)))**qcdparams
                
                    fail_qcd = rl.ParametericSample('ptbin%dmjjbin%d%sfail%s_qcd' % (ptbin, mjjbin, cat, year), rl.Sample.BACKGROUND, msd, scaledparams[0])
                    failCh.addSample(fail_qcd)
                    pass_qcd = rl.TransferFactorSample('ptbin%dmjjbin%d%spass%s_qcd' % (ptbin, mjjbin, cat, year), rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
                    passCh.addSample(pass_qcd)
                
                    failCh.mask = validbins[cat][ptbin]
                    passCh.mask = validbins[cat][ptbin]

            qcdfit_ws = ROOT.RooWorkspace('w')

            simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
            qcdfit = simpdf.fitTo(obs,
                                  ROOT.RooFit.Extended(True),
                                  ROOT.RooFit.SumW2Error(True),
                                  ROOT.RooFit.Strategy(2),
                                  ROOT.RooFit.Save(),
                                  ROOT.RooFit.Minimizer('Minuit2', 'migrad'),
                                  ROOT.RooFit.PrintLevel(1),
                              )
            qcdfit_ws.add(qcdfit)
            qcdfit_ws.writeToFile(os.path.join(str(tmpdir), 'testModel_qcdfit_'+cat+'_'+year+'.root'))

            # Set parameters to fitted values  
            allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
            pvalues = []
            for i, p in enumerate(tf_MCtempl.parameters.reshape(-1)):
                p.value = allparams[p.name]
                pvalues += [p.value]
            
            if qcdfit.status() != 0:
                print('Could not fit qcd')
                fitfailed_qcd += 1

                new_values = np.array(pvalues).reshape(tf_MCtempl.parameters.shape)
                with open("initial_vals_"+cat+".json", "w") as outfile:
                    json.dump({"initial_vals":new_values.tolist()},outfile)

            else:
                break

        if fitfailed_qcd >=5:
            raise RuntimeError('Could not fit qcd after 5 tries')

        print("Fitted qcd for category " + cat)

        # Plot the MC P/F transfer factor                                                   
        plot_mctf(tf_MCtempl,msdbins, cat)                           

        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + '_deco', qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)

        # initial values                                                                                                                                         
        with open('initial_vals_data_'+cat+'.json') as f:
            initial_vals_data = np.array(json.load(f)['initial_vals'])

        print("TFres order " + str(initial_vals_data.shape[0]-1)+ " in pT, " + str(initial_vals_data.shape[1]-1) + " in rho")
        tf_dataResidual = rl.BasisPoly("tf_dataResidual_"+year+cat,
                                       (initial_vals_data.shape[0]-1,initial_vals_data.shape[1]-1), 
                                       ['pt', 'rho'], 
                                       basis='Bernstein',
                                       init_params=initial_vals_data,
                                       limits=(-50,50), 
                                       coefficient_transform=None)

        tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
        tf_params[cat] = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params

    # build actual fit model now
    model = rl.Model('testModel_'+year)

    # exclude QCD from MC samps
    samps = ['ggF','VBF','WH','ZH','ttH','Wjets','Zjets','Zjetsbb','EWKW','EWKZ','ttbar','singlet','VV']
    sigs = ['ggF','VBF']

    for cat in cats:
        for ptbin in range(npt[cat]):
            for mjjbin in range(nmjj[cat]):
                for region in ['pass', 'fail']:

                    binindex = ptbin
                    if cat == 'vbf':
                        binindex = mjjbin

                    print("Bin: " + cat + " bin " + str(binindex) + " " + region)

                    # drop bins outside rho validity                                                
                    mask = validbins[cat][ptbin]
                    failCh.mask = mask
                    # blind bins 9-13                                                                        
                    mask[9:14] = False    
                    passCh.mask = mask

                    ch = rl.Channel('ptbin%dmjjbin%d%s%s%s' % (ptbin, mjjbin, cat, region, year))
                    model.addChannel(ch)

                    isPass = region == 'pass'
                    templates = {}
            
                    for sName in samps:

                        templates[sName] = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='nominal')
                        nominal = templates[sName][0]

                        if(badtemp_ma(nominal)):
                            print("Sample {} is too small, skipping".format(sName))
                            continue

                        # expectations
                        templ = templates[sName]
                        
                        if sName in sigs:
                            stype = rl.Sample.SIGNAL
                        else:
                            stype = rl.Sample.BACKGROUND
                    
                        MORPHNOMINAL = True
                        def smorph(templ):      
                            if templ is None:
                                return None                  
                            
                            if MORPHNOMINAL and sName not in ['QCD']:
                                return MorphHistW2(templ).get(shift=SF[year]['shift_SF']/smass('Wjets') * smass(sName),
                                                              smear=SF[year]['smear_SF']
                                )
                            else:
                                return templ
                        templ = smorph(templ)

                        sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                        # You need one systematic
                        sample.setParamEffect(sys_lumi_uncor, lumi[year[:4]]['uncorrelated'])
                        sample.setParamEffect(sys_lumi_cor_161718, lumi[year[:4]]['correlated'])
                        sample.setParamEffect(sys_lumi_cor_1718, lumi[year[:4]]['correlated_20172018'])

                        if do_systematics:

                            sample.autoMCStats(lnN=True)    

                            # Experimental systematics #######################################
                            
                            sample.setParamEffect(sys_eleveto, 1.005)
                            sample.setParamEffect(sys_muveto, 1.005)
                            sample.setParamEffect(sys_tauveto, 1.05)

                            for sys in exp_systs:

                                syst_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Up')[0]
                                syst_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Down')[0]

                                eff_up = shape_to_num(syst_up,nominal)
                                eff_do = shape_to_num(syst_do,nominal)

                                sample.setParamEffect(sys_dict[sys], eff_up, eff_do)

                            # Scale and Smear
                            mtempl = AffineMorphTemplate(templ)

                            if sName not in ['QCD']:
                                # shift
                                realshift = SF[year]['shift_SF_ERR']/smass('Wjets') * smass(sName)
                                _up = mtempl.get(shift=realshift)
                                _down = mtempl.get(shift=-realshift)
                                if badtemp_ma(_up[0]) or badtemp_ma(_down[0]):
                                    print("Skipping sample {}, scale systematic would be empty".format(sName))
                                else:
                                    sample.setParamEffect(sys_scale, _up, _down, scale=1)
                                        
                                    if cat == 'ggf':
                                        shiftpt = 3
                                        _up = mtempl.get(shift=shiftpt)
                                        _down = mtempl.get(shift=-1*shiftpt)
                                        sample.setParamEffect(sys_scale_pt[ptbin], _up, _down, scale=1)

                                # smear
                                _up = mtempl.get(smear=1 + SF[year]['smear_SF_ERR'])
                                _down = mtempl.get(smear=1 - SF[year]['smear_SF_ERR'])
                                if badtemp_ma(_up[0]) or badtemp_ma(_down[0]):
                                    print("Skipping sample {}, scale systematic would be empty".format(sName))
                                else:
                                    sample.setParamEffect(sys_smear, _up, _down)
                                    
                            # Muon CR phase space unc on ttbar                                                                                                             
                            if sName == "ttbar" and cat == "vbf":
                                sample.setParamEffect(sys_vbf_ttbar_unc,SF[year]["muCR_VBF_ttbar"])

                            # Theory systematics #########################################
                            # uncertainties on V+jets                 
                            if sName in ['Wjets']:
                                for sys in Wjets_thsysts:
                                    syst_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Up')[0]
                                    syst_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Down')[0]
                                    
                                    eff_up = shape_to_num(syst_up,nominal)
                                    eff_do = shape_to_num(syst_do,nominal)
                                    
                                    sample.setParamEffect(sys_dict[sys], eff_up, eff_do)

                            elif sName in ['Zjets','Zjetsbb']:
                                for sys in Zjets_thsysts:
                                    syst_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Up')[0]
                                    syst_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Down')[0]
                                    
                                    eff_up = shape_to_num(syst_up,nominal)
                                    eff_do = shape_to_num(syst_do,nominal)

                                    sample.setParamEffect(sys_dict[sys], eff_up, eff_do)

                            # QCD scale and PDF uncertainties on Higgs signal    
                            elif sName in ['ggF','VBF','WH','ZH','ggZH','ttH']:
                            
                                fsr_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='UEPS_FSRUp')[0]
                                fsr_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='UEPS_FSRDown')[0]
                                eff_fsr_up = np.sum(fsr_up)/np.sum(nominal)
                                eff_fsr_do = np.sum(fsr_do)/np.sum(nominal)

                                isr_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='UEPS_ISRUp')[0]
                                isr_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='UEPS_ISRDown')[0]
                                eff_isr_up = np.sum(isr_up)/np.sum(nominal)
                                eff_isr_do = np.sum(isr_do)/np.sum(nominal)

                                pdf_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='PDF_weightUp')[0]
                                pdf_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='PDF_weightDown')[0]
                                eff_pdf_up = np.sum(pdf_up)/np.sum(nominal)
                                eff_pdf_do = np.sum(pdf_do)/np.sum(nominal)
                            
                                if sName == 'ggF':
                                    scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_7ptUp')[0]
                                    scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_7ptDown')[0]
                                
                                    eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                    eff_scale_do = np.sum(scale_do)/np.sum(nominal)

                                    sample.setParamEffect(scale_ggF,eff_scale_up,eff_scale_do)
                                    sample.setParamEffect(pdf_Higgs_ggF,eff_pdf_up,eff_pdf_do)
                                    sample.setParamEffect(fsr_ggF,eff_fsr_up,eff_fsr_do)
                                    sample.setParamEffect(isr_ggF,eff_isr_up,eff_isr_do)

                                elif sName == 'VBF':
                                    scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptUp')[0]
                                    scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptDown')[0]

                                    eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                    eff_scale_do = np.sum(scale_do)/np.sum(nominal)

                                    sample.setParamEffect(scale_VBF,eff_scale_up,eff_scale_do)
                                    sample.setParamEffect(pdf_Higgs_VBF,eff_pdf_up,eff_pdf_do)
                                    sample.setParamEffect(fsr_VBF,eff_fsr_up,eff_fsr_do)
                                    sample.setParamEffect(isr_VBF,eff_isr_up,eff_isr_do)
                                    
                                elif sName in ['WH','ZH','ggZH']:
                                    scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptUp')[0]
                                    scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptDown')[0]

                                    eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                    eff_scale_do = np.sum(scale_do)/np.sum(nominal)

                                    if eff_scale_do < 0:
                                        eff_scale_do = eff_scale_up

                                    sample.setParamEffect(scale_VH,eff_scale_up,eff_scale_do)
                                    sample.setParamEffect(pdf_Higgs_VH,eff_pdf_up,eff_pdf_do)
                                    sample.setParamEffect(fsr_VH,eff_fsr_up,eff_fsr_do)
                                    sample.setParamEffect(isr_VH,eff_isr_up,eff_isr_do)

                                elif sName == 'ttH':
                                    scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_7ptUp')[0]
                                    scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_7ptDown')[0]

                                    eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                    eff_scale_do = np.sum(scale_do)/np.sum(nominal)

                                    sample.setParamEffect(scale_ttH,eff_scale_up,eff_scale_do)
                                    sample.setParamEffect(pdf_Higgs_ttH,eff_pdf_up,eff_pdf_do)
                                    sample.setParamEffect(fsr_ttH,eff_fsr_up,eff_fsr_do)
                                    sample.setParamEffect(isr_ttH,eff_isr_up,eff_isr_do)

                        # END if do_systematics  

                        # Add SFs last!
                        # DDB SF 
                        if sName in ['ggF','VBF','WH','ZH','ggZH','ttH','Zjetsbb']:
                            sf,sfunc_up,sfunc_down = passfailSF(isPass, sName, binindex, cat+'_', msd, mask, 1, SF[year]['BB_SF_UP'], SF[year]['BB_SF_DOWN'])
                            sample.scale(sf)
                            if do_systematics:
                                sample.setParamEffect(sys_ddxeffbb, sfunc_up, sfunc_down)
                                if cat == 'ggf':                                                                               
                                    sample.setParamEffect(sys_ddb_pt[ptbin],1.15)  

                        # N2DDT SF (V SF)                                                     
                        sample.scale(SF[year]['V_SF'])
                        if do_systematics:
                            effect = 1.0 + SF[year]['V_SF_ERR'] / SF[year]['V_SF']
                            sample.setParamEffect(sys_veff,effect)
                            if cat == 'ggf':
                                sample.setParamEffect(sys_veff_pt[ptbin],1.10)

                        ch.addSample(sample)
 
                    # END loop over MC samples 

                    data_obs = get_template('data', isPass, binindex+1, cat+'_', obs=msd, syst='nominal')

                    if not isPass:
                        Nfail_data += data_obs[0].sum()

                    ch.setObservation(data_obs, read_sumw2=True)

    for cat in cats:
        for ptbin in range(npt[cat]):
            for mjjbin in range(nmjj[cat]):

                failCh = model['ptbin%dmjjbin%d%sfail%s' % (ptbin, mjjbin, cat, year)]
                passCh = model['ptbin%dmjjbin%d%spass%s' % (ptbin, mjjbin, cat, year)]

                qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%dmjjbin%d%s%s_%d' % (ptbin, mjjbin, cat, year, i), 0, -50, 50) for i in range(msd.nbins)])
                initial_qcd = failCh.getObservation()[0].astype(float)  # was integer, and numpy complained about subtracting float from it

                for sample in failCh:
                    initial_qcd -= sample.getExpectation(nominal=True)

                if np.any(initial_qcd < 0.):
                    initial_qcd[np.where(initial_qcd<0)] = 0
                    #raise ValueError('initial_qcd negative for some bins..', initial_qcd)

                sigmascale = 10  # to scale the deviation from initial                      
                scaledparams = initial_qcd * (1 + sigmascale/np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
                fail_qcd = rl.ParametericSample('ptbin%dmjjbin%d%sfail%s_qcd' % (ptbin, mjjbin, cat, year), rl.Sample.BACKGROUND, msd, scaledparams)
                failCh.addSample(fail_qcd)
                pass_qcd = rl.TransferFactorSample('ptbin%dmjjbin%d%spass%s_qcd' % (ptbin, mjjbin, cat, year), rl.Sample.BACKGROUND, tf_params[cat][ptbin, :], fail_qcd)
                passCh.addSample(pass_qcd)

                if do_muon_CR:
                
                    tqqpass = passCh['ttbar']
                    tqqfail = failCh['ttbar']
                    sumPass = tqqpass.getExpectation(nominal=True).sum()
                    sumFail = tqqfail.getExpectation(nominal=True).sum()

                    if 'singlet' in passCh.samples:
                        stqqpass = passCh['singlet']
                        stqqfail = failCh['singlet']
                    
                        sumPass += stqqpass.getExpectation(nominal=True).sum()
                        sumFail += stqqfail.getExpectation(nominal=True).sum()

                        tqqPF =  sumPass / sumFail

                        stqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                        stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
                        stqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                        stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

                    tqqPF =  sumPass / sumFail
                    tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
                    tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                    tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                    
    kfactor_qcd = 1.0*Nfail_data/Nfail_qcd_MC

    # Fill in muon CR
    if do_muon_CR:
        templates = {}
        samps = ['QCD','Wjets','Zjets','Zjetsbb','ttbar','singlet']
        for region in ['pass', 'fail']:
            ch = rl.Channel('muonCR%s%s' % (region, year))
            model.addChannel(ch)

            isPass = region == 'pass'
            print("Bin: muon cr " + region)

            for sName in samps:

                templates[sName] = one_bin(sName, isPass, -1, '', obs=msd, syst='nominal', muon=True)
                nominal = templates[sName][0]

                if nominal < eps:
                    print("Sample {} is too small, skipping".format(sName))
                    continue

                stype = rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templates[sName])

                # You need one systematic
                sample.setParamEffect(sys_lumi_uncor, lumi[year[:4]]['uncorrelated'])
                sample.setParamEffect(sys_lumi_cor_161718, lumi[year[:4]]['correlated'])
                sample.setParamEffect(sys_lumi_cor_1718, lumi[year[:4]]['correlated_20172018'])

                if do_systematics:

                    sample.autoMCStats(lnN=True)

                    sample.setParamEffect(sys_eleveto, 1.005)
                    sample.setParamEffect(sys_tauveto, 1.05)

                    # End of systematics applied to QCD
                    if sName == 'QCD':
                        ch.addSample(sample)
                        continue

                    for sys in mu_exp_systs:
                        syst_up = one_bin(sName, isPass, -1, '', obs=msd, syst=sys+'Up', muon=True)[0]
                        syst_do = one_bin(sName, isPass, -1, '', obs=msd, syst=sys+'Down', muon=True)[0]
                
                        eff_up = shape_to_num(syst_up,nominal)
                        eff_do = shape_to_num(syst_do,nominal)

                        sample.setParamEffect(sys_dict[sys], eff_up, eff_do)  


#                        if 'bc' in sys and '2017' in year:
#                            if abs(eff_up-1) > abs(eff_do-1):
#                                sample.setParamEffect(sys_dict[sys], eff_up)
#                            else:
#                                sample.setParamEffect(sys_dict[sys], eff_do)
#                        elif 'pileup' in sys and '2017' in year:
#                            print(sys, eff_up, eff_do)
#                        else:
#sample.setParamEffect(sys_dict[sys], eff_up, eff_do)

                # END if do_systematics 

                # DDB SF                                                                                  
                if sName in ['ggF','VBF','WH','ZH','ggZH','ttH','Zjetsbb']:
                    sf,sfunc_up,sfunc_down = passfailSF(isPass, sName, -1, '', msd, mask, 1, SF[year]['BB_SF_UP'], SF[year]['BB_SF_DOWN'], muon = True)
                    sample.scale(sf)
                    if do_systematics:
                        sample.setParamEffect(sys_ddxeffbb, sfunc_up, sfunc_down)

                # N2DDT SF (V SF)                                                            
                sample.scale(SF[year]['V_SF'])
                if do_systematics:
                    effect = 1.0 + SF[year]['V_SF_ERR'] / SF[year]['V_SF']
                    sample.setParamEffect(sys_veff,effect)

                ch.addSample(sample)

            # END loop over MC samples

            data_obs = one_bin('muondata', isPass, -1, '', obs=msd, syst='nominal', muon=True)
            ch.setObservation(data_obs, read_sumw2=True)

        tqqpass = model['muonCRpass'+year+'_ttbar']
        tqqfail = model['muonCRfail'+year+'_ttbar']
        sumPass = tqqpass.getExpectation(nominal=True).sum()
        sumFail = tqqfail.getExpectation(nominal=True).sum()

        stqqpass = model['muonCRpass'+year+'_singlet']
        stqqfail = model['muonCRfail'+year+'_singlet']
        sumPass += stqqpass.getExpectation(nominal=True).sum()
        sumFail += stqqfail.getExpectation(nominal=True).sum()

        tqqPF =  sumPass / sumFail

        stqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        stqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        
        tqqpass.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
        tqqpass.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

        # END if do_muon_CR  

    with open(os.path.join(str(tmpdir), 'testModel_'+year+'.pkl'), 'wb') as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), 'testModel_'+year))

if __name__ == '__main__':

    year = "2016"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    elif "2017" in thisdir: 
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    print("Running for "+year)

    if not os.path.exists('output'):
        os.mkdir('output')

    ggfvbf_rhalphabet('output',year)

