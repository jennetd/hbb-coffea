from __future__ import print_function, division
import sys, os
import csv, json
import numpy as np
import ROOT
import pandas as pd

from make_cards import *

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
    print(name)

    h = f.Get(name)

    sumw = []
    sumw2 = []

    for i in range(1,h.GetNbinsX()+1):
        sumw += [h.GetBinContent(i)]
        sumw2 += [h.GetBinError(i)*h.GetBinError(i)]

    return (np.array(sumw), obs.binning, obs.name, np.array(sumw2))

def shape_to_num(var, nom, clip=2):
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

def systematics(year):

    with open('sf.json') as f:
        SF = json.load(f)

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

    exp_systs = ['jet_trigger','pileup_weight',
                 'JES','JER','UES',
                 'btagSFlight_'+year,'btagSFbc_'+year,
                 'btagSFlight_correlated','btagSFbc_correlated']

    if '2018' not in year:
        exp_systs += ['L1Prefiring']
    mu_exp_systs = exp_systs + ['muon_ID_'+yearstr+'_value','muon_ISO_'+yearstr+'_value','muon_TRIGNOISO_'+yearstr+'_value']

    Zjets_thsysts = ['d1kappa_EW', 'Z_d2kappa_EW', 'Z_d3kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO']
    Wjets_thsysts = ['d1kappa_EW', 'W_d2kappa_EW', 'W_d3kappa_EW', 'd1K_NLO', 'd2K_NLO', 'd3K_NLO']         
                      
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

    cats = ['ggf','vbf']
    ncat = len(cats)

    # exclude QCD from MC samps
    samps = ['ggF','VBF','WH','ZH','ttH','ttbar','singlet','Zjets','Zjetsbb','Wjets','VV']

    cols = ['bin','region','samp','syst','up','val']
    df = pd.DataFrame(columns=cols)

    for cat in cats:

        ptpts, msdpts = np.meshgrid(ptbins[cat][:-1] + 0.3 * np.diff(ptbins[cat]), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
        rhopts = 2*np.log(msdpts/ptpts)
        ptscaled = (ptpts - 450.) / (1200. - 450.)
        rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
        validbins[cat] = (rhoscaled >= 0) & (rhoscaled <= 1)
        rhoscaled[~validbins[cat]] = 1  # we will mask these out later   

        for ptbin in range(npt[cat]):
            for mjjbin in range(nmjj[cat]):
                for region in ['pass', 'fail']:

                    binindex = ptbin
                    if cat == 'vbf':
                        binindex = mjjbin

                    print("Bin: " + cat + " bin " + str(binindex) + " " + region)
                    mask = validbins[cat][ptbin]

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
                        
                        for sys in exp_systs:
                            syst_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Up')[0]
                            syst_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Down')[0]

                            eff_up = shape_to_num(syst_up,nominal)
                            eff_do = shape_to_num(syst_do,nominal)

                            if "btagSFlight_201" in sys or "btagSFbc_201" in sys:
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys.split('_')[0],1,eff_up-1]],columns=cols))
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys.split('_')[0],0,eff_do-1]],columns=cols))
                            else:
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys,1,eff_up-1]],columns=cols))
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys,0,eff_do-1]],columns=cols))

                        # Theory systematics #########################################
                        # uncertainties on V+jets                 
                        if sName in ['Wjets']:
                            for sys in Wjets_thsysts:
                                syst_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Up')[0]
                                syst_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Down')[0]
                                
                                eff_up = shape_to_num(syst_up,nominal)
                                eff_do = shape_to_num(syst_do,nominal)
                                    
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys,1,eff_up-1]],columns=cols))
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys,0,eff_do-1]],columns=cols))

                        elif sName in ['Zjets','Zjetsbb']:
                            for sys in Zjets_thsysts:
                                syst_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Up')[0]
                                syst_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst=sys+'Down')[0]
                                    
                                eff_up = shape_to_num(syst_up,nominal)
                                eff_do = shape_to_num(syst_do,nominal)

                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys,1,eff_up-1]],columns=cols))
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,sys,0,eff_do-1]],columns=cols))

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
                                
                            elif sName == 'VBF':
                                scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptUp')[0]
                                scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptDown')[0]
                                
                                eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                eff_scale_do = np.sum(scale_do)/np.sum(nominal)
                                
                            elif sName in ['WH','ZH','ggZH']:
                                scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptUp')[0]
                                scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_3ptDown')[0]
                                
                                eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                eff_scale_do = np.sum(scale_do)/np.sum(nominal)
                                
                                if eff_scale_do < 0:
                                    eff_scale_do = eff_scale_up
                                    
                            elif sName == 'ttH':
                                scale_up = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_7ptUp')[0]
                                scale_do = get_template(sName, isPass, binindex+1, cat+'_', obs=msd, syst='scalevar_7ptDown')[0]
                                
                                eff_scale_up = np.sum(scale_up)/np.sum(nominal)
                                eff_scale_do = np.sum(scale_do)/np.sum(nominal)
                                    
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'scale',1,eff_scale_up-1]],columns=cols))
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'PDF',1,eff_pdf_up-1]],columns=cols))
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'FSR',1,eff_fsr_up-1]],columns=cols))
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'ISR',1,eff_isr_up-1]],columns=cols))
                            
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'scale',0,eff_scale_do-1]],columns=cols))
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'PDF',0,eff_pdf_do-1]],columns=cols))
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'FSR',0,eff_fsr_do-1]],columns=cols))
                            df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'ISR',0,eff_isr_do-1]],columns=cols))
                        
                        # Add SFs last!
                        # DDB SF 
                        if sName in ['ggF','VBF','WH','ZH','ggZH','ttH','Zjetsbb']:
                                sf,sfunc_up,sfunc_down = passfailSF(isPass, sName, binindex, cat+'_', msd, mask, 1, SF[year]['BB_SF_UP'], SF[year]['BB_SF_DOWN'])

                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'DDB',1,sfunc_up-1]],columns=cols))
                                df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'DDB',0,sfunc_down-1]],columns=cols))

                        # N2DDT SF (V SF)                                                     
                        effect = 1.0 + SF[year]['V_SF_ERR'] / SF[year]['V_SF']
                        df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'N2DDT',1,effect-1]],columns=cols))
                        df = df.append(pd.DataFrame([[cat+' '+str(binindex+1),region,sName,'N2DDT',0,effect-1]],columns=cols))

    df.to_csv('systematics.csv')

if __name__ == '__main__':

    year = "2016"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    if "2017" in thisdir: 
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    print("Running for "+year)

    systematics(year)

