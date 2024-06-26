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
rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False

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
        sumw += [h.GetBinContent(i)]
        sumw2 += [h.GetBinError(i)*h.GetBinError(i)]

    return (np.array(sumw), obs.binning, obs.name, np.array(sumw2))

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

    # define bins    
    ptbins = {}
    ptbins['ggf'] = np.array([450, 500, 550, 600, 675, 800, 1200])
    ptbins['vbf'] = np.array([450,1200])

    mjjbins = {}
    mjjbins['ggf'] = np.array([0,4000])
    mjjbins['vbf'] = np.array([1000,2000,4000])

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

                    # QCD templates from file                           
                    if cat == "vbf":
                        failTempl = get_template('QCD', 0, mjjbin+1, cat+'_', obs=msd, syst='nominal')
                        passTempl = get_template('QCD', 1, mjjbin+1, cat+'_', obs=msd, syst='nominal')
                    else: 
                        failTempl = get_template('QCD', 0, ptbin+1, cat+'_', obs=msd, syst='nominal')
                        passTempl = get_template('QCD', 1, ptbin+1, cat+'_', obs=msd, syst='nominal')

                    failCh.setObservation(failTempl, read_sumw2=True)
                    passCh.setObservation(passTempl, read_sumw2=True)

                    qcdfail += sum([val for val in failCh.getObservation()[0]])
                    qcdpass += sum([val for val in passCh.getObservation()[0]])

            qcdeff = qcdpass / qcdfail
            print('Inclusive P/F from Monte Carlo = ' + str(qcdeff))

            # initial values                                                                 
            print('Initial fit values read from file initial_vals*')
            with open('initial_vals_'+cat+'.json') as f:
                initial_vals = np.array(json.load(f)['initial_vals'])
            print(initial_vals)

            tf_MCtempl = rl.BasisPoly("tf_MCtempl_"+cat+year,
                                      (initial_vals.shape[0]-1,initial_vals.shape[1]-1),
                                      ['pt', 'rho'],
                                      basis='Bernstein',
                                      init_params=initial_vals,
                                      limits=(-10, 10), coefficient_transform=None)
            tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)

            for ptbin in range(npt[cat]):
                for mjjbin in range(nmjj[cat]):

                    failCh = qcdmodel['ptbin%dmjjbin%d%sfail%s' % (ptbin, mjjbin, cat, year)]
                    passCh = qcdmodel['ptbin%dmjjbin%d%spass%s' % (ptbin, mjjbin, cat, year)]
                    failObs = failCh.getObservation()
                    passObs = passCh.getObservation()
                
                    qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%dmjjbin%d%s%s_%d' % (ptbin, mjjbin, cat, year, i), 0) for i in range(msd.nbins)])
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
#        plot_mctf(tf_MCtempl,msdbins, cat)                           

        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + '_deco', qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)

        tf_dataResidual = rl.BasisPoly("tf_dataResidual_"+year+cat,
                                       (0,0),
                                       ['pt', 'rho'],
                                       basis='Bernstein',
                                       init_params=np.array([[1]]),
                                       limits=(-20,20),
                                       coefficient_transform=None)
        tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
        tf_params[cat] = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params

    # build actual fit model now
    model = rl.Model('testModel_'+year)

    # exclude QCD from MC samps
    samps = ['ggF']
    sigs = ['ggF']

    for cat in cats:
        for ptbin in range(npt[cat]):
            for mjjbin in range(nmjj[cat]):
                for region in ['pass', 'fail']:

                    binindex = ptbin
                    if cat == 'vbf':
                        binindex = mjjbin

                    # drop bins outside rho validity                                                
                    mask = validbins[cat][ptbin]

                    ch = rl.Channel('ptbin%dmjjbin%d%s%s%s' % (ptbin, mjjbin, cat, region, year))
                    model.addChannel(ch)

                    isPass = region == 'pass'
                    templates = {}
            
                    for sName in samps:

                        templates[sName] = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='nominal')
                        nominal = templates[sName][0]

                        # expectations
                        templ = templates[sName]
                        
                        if sName in sigs:
                            stype = rl.Sample.SIGNAL
                        else:
                            stype = rl.Sample.BACKGROUND
                    
                        sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)
                
                        ch.addSample(sample)

                    # Observed data = QCD MC
                    data_obs = get_template('QCD', isPass, binindex+1, cat+'_', obs=msd, syst='nominal')

                    ch.setObservation(data_obs, read_sumw2=True)

                    # drop bins outside rho validity
                    mask = validbins[cat][ptbin]

    for cat in cats:
        for ptbin in range(npt[cat]):
            for mjjbin in range(nmjj[cat]):

                failCh = model['ptbin%dmjjbin%d%sfail%s' % (ptbin, mjjbin, cat, year)]
                passCh = model['ptbin%dmjjbin%d%spass%s' % (ptbin, mjjbin, cat, year)]

                qcdparams = np.array([rl.IndependentParameter('qcdparam_ptbin%dmjjbin%d%s%s_%d' % (ptbin, mjjbin, cat, year, i), 0) for i in range(msd.nbins)])
                initial_qcd = failCh.getObservation()[0].astype(float)  # was integer, and numpy complained about subtracting float from it                                
                if np.any(initial_qcd < 0.):
                    raise ValueError('initial_qcd negative for some bins..', initial_qcd)

                sigmascale = 10  # to scale the deviation from initial                                                                                                     
                scaledparams = initial_qcd * (1 + sigmascale/np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
                fail_qcd = rl.ParametericSample('ptbin%dmjjbin%d%sfail%s_qcd' % (ptbin, mjjbin, cat, year), rl.Sample.BACKGROUND, msd, scaledparams)
                failCh.addSample(fail_qcd)
                pass_qcd = rl.TransferFactorSample('ptbin%dmjjbin%d%spass%s_qcd' % (ptbin, mjjbin, cat, year), rl.Sample.BACKGROUND, tf_params[cat][ptbin, :], fail_qcd)
                passCh.addSample(pass_qcd)

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

