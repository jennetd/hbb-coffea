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

eps=0.0000001
do_muon_CR = False

######----------------HELPER FUNCTIONS-----------------#######
# Tell me if my sample is too small to care about
def badtemp_ma(hvalues, mask=None):
    # Need minimum size & more than 1 non-zero bins                                                                                                                                   
    tot = np.sum(hvalues[mask])

    count_nonzeros = np.sum(hvalues[mask] > 0)
    if (tot < eps) or (count_nonzeros < 2):
        return True
    else:
        return False

# Turn an msd distribution into a single bin (for muon CR)
def one_bin(template):
    try:
        h_vals, h_edges, h_key, h_variances = template
        return (np.array([np.sum(h_vals)]), np.array([0., 1.]), "onebin", np.array([np.sum(h_variances)]))
    except:
        h_vals, h_edges, h_key = template
        return (np.array([np.sum(h_vals)]), np.array([0., 1.]), "onebin")

# Read the histogram
def get_template(sName, passed, ptbin, cat, obs, syst, muon=False):
    """
    Read msd template from root file
    """

    f = ROOT.TFile.Open('{}/signalregion.root'.format(year))

    if muon:
        f = ROOT.TFile.Open('{}/muonCR.root'.format(year))

    #Determind the right branch
    name = 'fail_mv1_'

    if passed:
        name = 'pass_mv1_'

    if cat == "charm":
        name = 'c_' + name
    elif cat == 'light':
        name = 'l_' + name

    name += sName+'_'+syst

    print("Extracting ... ", name)
    h = f.Get(name)

    sumw = []
    sumw2 = []

    for i in range(1,h.GetNbinsX()+1):
        sumw += [h.GetBinContent(i)]
        sumw2 += [h.GetBinError(i)*h.GetBinError(i)]

    return (np.array(sumw)[1:], obs.binning, obs.name, np.array(sumw2)[1:])

def vh_rhalphabet(tmpdir, throwPoisson = True, fast=0):
    """ 
    Create the data cards!

    1. Fit QCD MC Only
    2. Fill in actual fit model to every thing except for QCD
    3. Fill QCD in the actual fit model
    """
    with open('lumi.json') as f:
        lumi = json.load(f)

    ###>>>>>> TODO: DIFFERENT NUMBERS FOR VH???
    #Depends on how we define the ttbar control region.
    #If we define a control region in VH, it might be the same control region
    vbf_ttbar_unc = dict({"2016":1.29,"2017":1.62,"2018":1.52})

    # TT params #TODO: WHAT IS THIS?
    # Scale factor from the muon control region
    # Allowed to float
    # not signal strength: nuisance parameter. 
    tqqeffSF = rl.IndependentParameter('tqqeffSF_{}'.format(year), 1., -10, 50) #ddb efficiency
    tqqnormSF = rl.IndependentParameter('tqqnormSF_{}'.format(year), 1., -10, 50) #Overall scale factor on ttbar

    # Simple lumi systematics
    # Changes event yield
    # Constraints applied to overall likelihood                                                                                                                                                        
    sys_lumi_uncor = rl.NuisanceParameter('CMS_lumi_13TeV_{}'.format(year), 'lnN') #lnN: Log Normal
    sys_lumi_cor_161718 = rl.NuisanceParameter('CMS_lumi_13TeV_correlated_', 'lnN')
    sys_lumi_cor_1718 = rl.NuisanceParameter('CMS_lumi_13TeV_correlated_20172018', 'lnN')

    # define bins TODO: REDEFINE BINS??
    # In VH
    # Charm category and light category
    # Within each can have bins in pt and other variables like mass
    # Start with two category and no differential bins.
    ptbins = {}
    ptbins['charm'] = np.array([450,1200])
    ptbins['light'] = np.array([450,1200])

    npt = {}
    npt['charm'] = len(ptbins['charm']) - 1
    npt['light'] = len(ptbins['light']) - 1

    msdbins = np.linspace(47, 201, 23)
    msd = rl.Observable('msd', msdbins)

    validbins = {}

    cats = ['light', 'charm'] #TODO: Charm and light. Might consider 'ZH', 'WH'
    ncat = len(cats)

    # Build qcd MC pass+fail model and fit to polynomial
    # QCD MC only fit
    tf_params = {}
    for cat in cats:

        fitfailed_qcd = 0

        # here we derive these all at once with 2D array  
        #>>>>>>>                          
        ptpts, msdpts = np.meshgrid(ptbins[cat][:-1] + 0.3 * np.diff(ptbins[cat]), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
        rhopts = 2*np.log(msdpts/ptpts) 
        ptscaled = (ptpts - 450.) / (1200. - 450.) #Keep same with vbf
        rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6)) 

        validbins[cat] = (rhoscaled >= 0) & (rhoscaled <= 1)
        rhoscaled[~validbins[cat]] = 1  # we will mask these out later   

        while fitfailed_qcd < 5: #Fail if choose bad initial values, start from where the fits fail. 
        
            qcdmodel = rl.Model('qcdmodel_'+cat)
            qcdpass, qcdfail = 0., 0.

            ##>>>>>>>!!!Cut out this for loop?
            for ptbin in range(npt[cat]):

                failCh = rl.Channel('ptbin%d%s%s%s' % (ptbin, cat, 'fail',year))
                passCh = rl.Channel('ptbin%d%s%s%s' % (ptbin, cat, 'pass',year))
                qcdmodel.addChannel(failCh)
                qcdmodel.addChannel(passCh)

                binindex = ptbin

                # QCD templates from file
                # Look at the root file and see what's in there.                         
                failTempl = get_template('QCD', 0, binindex+1, cat, obs=msd, syst='nominal')
                passTempl = get_template('QCD', 1, binindex+1, cat, obs=msd, syst='nominal')

                failCh.setObservation(failTempl, read_sumw2=True)
                passCh.setObservation(passTempl, read_sumw2=True)

                qcdfail += sum([val for val in failCh.getObservation()[0]])
                qcdpass += sum([val for val in passCh.getObservation()[0]])

            qcdeff = qcdpass / qcdfail
            print('Inclusive P/F from Monte Carlo = ' + str(qcdeff))

            # initial values
            # Right size of arrays
            # Want to make an zeroth order in pt
            # {"initial_vals":[[1,1]]} in json file (0th pt and 1st in rho)                                                               
            print('Initial fit values read from file initial_vals*')
            with open('initial_vals_'+cat+'.json') as f:
                initial_vals = np.array(json.load(f)['initial_vals'])
            print(initial_vals)

            #tf polynomial
            tf_MCtempl = rl.BasisPoly("tf_MCtempl_"+cat+year, #name
                                      (initial_vals.shape[0]-1,initial_vals.shape[1]-1), #shape
                                      ['pt', 'rho'], #variable names
                                      basis='Bernstein', #type of polys
                                      init_params=initial_vals, #initial values
                                      limits=(-10, 10), #limits on poly coefficients
                                      coefficient_transform=None)

            tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled) #make sure the coeff stays of order 1. 

            for ptbin in range(npt[cat]):
                    
                    #!!! I guess we need to cut out the loop but still need to keep some of the code to
                    #fit within the category
                    failCh = qcdmodel['ptbin%d%sfail%s' % (ptbin, cat, year)]
                    passCh = qcdmodel['ptbin%d%spass%s' % (ptbin, cat, year)]
                    failObs = failCh.getObservation()
                    passObs = passCh.getObservation()
                
                    qcdparams = np.array([rl.IndependentParameter('qcdparam_'+cat+'_ptbin%d' % ptbin, 0)])
                    sigmascale = 10.
                    scaledparams = failObs * (1 + sigmascale/np.maximum(1., np.sqrt(failObs)))**qcdparams
                
                    fail_qcd = rl.ParametericSample('ptbin%d%sfail%s_qcd' % (ptbin, cat, year), rl.Sample.BACKGROUND, msd, scaledparams[0])
                    failCh.addSample(fail_qcd)
                    pass_qcd = rl.TransferFactorSample('ptbin%d%spass%s_qcd' % (ptbin, cat, year), rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin, :], fail_qcd)
                    passCh.addSample(pass_qcd)

                    #MC only fit, don't need to blind higgs region. 
                    failCh.mask = validbins[cat][ptbin] 
                    passCh.mask = validbins[cat][ptbin]

            #Run the fit for qcd mc pass/fail
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
            # Check what the values and if the fit fails
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

        print("Fitted qcd for category " + cat) #Know the whole part works

        # Plot the MC P/F transfer factor                                                   
        # plot_mctf(tf_MCtempl,msdbins, cat)                           

        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + '_deco', qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
        tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)

        # initial values                                                                                                            # for a different tf, can just copy the other ones in a
        # different file                        
        with open('initial_vals_data_'+cat+'.json') as f:
            initial_vals_data = np.array(json.load(f)['initial_vals'])

        # Fitting ratio of the data and the MC prediction
        tf_dataResidual = rl.BasisPoly("tf_dataResidual_"+year+cat,
                                       (initial_vals_data.shape[0]-1,initial_vals_data.shape[1]-1), 
                                       ['pt', 'rho'],
                                       basis='Bernstein',
                                       init_params=initial_vals_data,
                                       limits=(-20,20), 
                                       coefficient_transform=None)

        tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
        tf_params[cat] = qcdeff * tf_MCtempl_params_final * tf_dataResidual_params

    # build actual fit model now
    # the model that would go into the workspace.
    model = rl.Model('testModel_'+year)

    # exclude QCD from MC samps
    samps = ['ggF','VBF','WH','ZH','ttH','ttbar','singlet','Zjets','Zjetsbb','Wjets','VV']
    sigs = ['ZH','WH']

    #Fill actual fit model with the expected fit value for every process except for QCD
    # Model need to know the signal, and background
    # different background have different uncertainties
    # Don't treat the QCD like everything else
    # Take the QCD expectation from previous fit.
    # Take expectation from MC
    for cat in cats:
        for ptbin in range(npt[cat]):
                for region in ['pass', 'fail']: #Separate also by b scores in addition to charm scores.

                    binindex = ptbin

                    print("Bin: " + cat + " bin " + str(binindex) + " " + region)

                    # drop bins outside rho validity                                                
                    mask = validbins[cat][ptbin] # If want to can blind the higgs peak to not confuse the model.
                    # mask[9:14] = False to blind, don't need to do it right now since using -t -1
                    failCh.mask = validbins[cat][ptbin]
                    passCh.mask = validbins[cat][ptbin]

                    ch = rl.Channel('ptbin%d%s%s%s' % (ptbin, cat, region, year))
                    model.addChannel(ch)

                    isPass = region == 'pass'
                    templates = {}
            
                    for sName in samps:

                        templates[sName] = get_template(sName, isPass, binindex+1, cat, obs=msd, syst='nominal')
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
                    
                        sample = rl.TemplateSample(ch.name + '_' + sName, stype, templ)

                        # You need one systematic
                        sample.setParamEffect(sys_lumi_uncor, lumi[year]['uncorrelated'])
                        sample.setParamEffect(sys_lumi_cor_161718, lumi[year]['correlated'])
                        sample.setParamEffect(sys_lumi_cor_1718, lumi[year]['correlated_20172018'])

                        ch.addSample(sample)

                    data_obs = get_template('data', isPass, binindex+1, cat, obs=msd, syst='nominal')

                    ch.setObservation(data_obs, read_sumw2=True)

    #Fill in the QCD in the actual fit model. 
    for cat in cats:
        for ptbin in range(npt[cat]):

                failCh = model['ptbin%d%sfail%s' % (ptbin, cat, year)]
                passCh = model['ptbin%d%spass%s' % (ptbin, cat, year)]

                qcdparams = np.array([rl.IndependentParameter('qcdparam_'+cat+'_ptbin%d' % (ptbin), 0)])
                initial_qcd = failCh.getObservation()[0].astype(float)  # was integer, and numpy complained about subtracting float from it

                for sample in failCh:
                    #Subtract away from data all mc processes except for QCD
                    initial_qcd -= sample.getExpectation(nominal=True)

                if np.any(initial_qcd < 0.):
                    raise ValueError('initial_qcd negative for some bins..', initial_qcd)

                sigmascale = 10  # to scale the deviation from initial                      
                scaledparams = initial_qcd * (1 + sigmascale/np.maximum(1., np.sqrt(initial_qcd)))**qcdparams
                fail_qcd = rl.ParametericSample('ptbin%d%sfail%s_qcd' % (ptbin, cat, year), rl.Sample.BACKGROUND, msd, scaledparams)
                failCh.addSample(fail_qcd)
                pass_qcd = rl.TransferFactorSample('ptbin%d%spass%s_qcd' % (ptbin, cat, year), rl.Sample.BACKGROUND, tf_params[cat][ptbin, :], fail_qcd)
                passCh.addSample(pass_qcd)

                if do_muon_CR:
                
                    tqqpass = passCh['ttbar']
                    tqqfail = failCh['ttbar']
                    tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
                    tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
                    tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
                    tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
                    tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)

    # Fill in muon CR
    # Ingore this for now. 
    # if do_muon_CR:
    #     templates = {}
    #     samps = ['ttbar','QCD','singlet','Zjets','Zjetsbb','Wjets','VV']
    #     for region in ['pass', 'fail']:
    #         ch = rl.Channel('muonCR%s%s' % (region, year))
    #         model.addChannel(ch)

    #         isPass = region == 'pass'
    #         print("Bin: muon cr " + region)

    #         for sName in samps:

    #             templates[sName] = one_bin(get_template(sName, isPass, -1, '', obs=msd, syst='nominal', muon=True))
    #             nominal = templates[sName][0]

    #             if(np.sum(nominal) < eps):
    #                 print("Sample {} is too small, skipping".format(sName))
    #                 continue

    #             stype = rl.Sample.BACKGROUND
    #             sample = rl.TemplateSample(ch.name + '_' + sName, stype, templates[sName])

    #             # You need one systematic
    #             sample.setParamEffect(sys_lumi_uncor, lumi[year]['uncorrelated'])
    #             sample.setParamEffect(sys_lumi_cor_161718, lumi[year]['correlated'])
    #             sample.setParamEffect(sys_lumi_cor_1718, lumi[year]['correlated_20172018'])

    #             ch.addSample(sample)

    #         # END loop over MC samples

    #         data_obs = one_bin(get_template('muondata', isPass, -1, '', obs=msd, syst='nominal', muon=True))
    #         ch.setObservation(data_obs, read_sumw2=True)

    #     tqqpass = model['muonCRpass'+year+'_ttbar']
    #     tqqfail = model['muonCRfail'+year+'_ttbar']
    #     tqqPF = tqqpass.getExpectation(nominal=True).sum() / tqqfail.getExpectation(nominal=True).sum()
    #     tqqpass.setParamEffect(tqqeffSF, 1*tqqeffSF)
    #     tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)
    #     tqqpass.setParamEffect(tqqnormSF, 1*tqqnormSF)
    #     tqqfail.setParamEffect(tqqnormSF, 1*tqqnormSF)
        
        # END if do_muon_CR  

    with open(os.path.join(str(tmpdir), 'testModel_'+year+'.pkl'), 'wb') as fout:
        pickle.dump(model, fout)

    model.renderCombine(os.path.join(str(tmpdir), 'testModel_'+year))


def main():

    #Setting different years depending on 
    if len(sys.argv) < 2:
        print("Enter year")
        return

    global year
    year = sys.argv[1]

    print("Running for " + year)

    outdir = '{}/output'.format(year)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    vh_rhalphabet(outdir, year)

if __name__ == '__main__':

    year = ""

    main()