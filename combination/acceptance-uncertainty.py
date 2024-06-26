# ACCEPTANCE
import json
import sys
import numpy as np
import pickle
import pandas as pd
import json

def acceptance_uncertainty(acc_unc, reco,true,syst):
    
    acc_up = pd.DataFrame()
    acc_do = pd.DataFrame()
    
    nominal = pd.DataFrame()
    reco_up = pd.DataFrame()
    reco_do = pd.DataFrame()
    
    # Loop over H processes ("mode" in hists)
    for i in (1,8): # mode
            
        nominal = pd.DataFrame.from_dict(reco.integrate('mode',int_range=slice(i,i+1)).integrate('systematic','nominal').values())
        reco_up = pd.DataFrame.from_dict(reco.integrate('mode',int_range=slice(i,i+1)).integrate('systematic',syst+'Up').values())
        reco_do = pd.DataFrame.from_dict(reco.integrate('mode',int_range=slice(i,i+1)).integrate('systematic',syst+'Down').values())

        nominal['true'] = pd.DataFrame.from_dict(true.integrate('mode',int_range=slice(i,i+1)).integrate('systematic','nominal').values()[()])
        reco_up['true'] = pd.DataFrame.from_dict(true.integrate('mode',int_range=slice(i,i+1)).integrate('systematic',syst+'Up').values()[()])
        reco_do['true'] = pd.DataFrame.from_dict(true.integrate('mode',int_range=slice(i,i+1)).integrate('systematic',syst+'Down').values()[()])
        
        if nominal.sum(axis=1).sum(axis=0) == 0:
            continue

        mode = 100 * i
        
        frac_up = (reco_up/nominal)
        frac_do = (reco_do/nominal)
        
        df_up = pd.DataFrame()
        df_do = pd.DataFrame()
        
        cats = [c for c in nominal.columns if 'muon' not in c[0]]
        for c in cats:
            df_up[c] = np.nan_to_num(np.divide(frac_up[c],frac_up['true'].to_numpy().flatten()))
            df_do[c] = np.nan_to_num(np.divide(frac_do[c],frac_do['true'].to_numpy().flatten()))

        df_up.set_index(df_up.index+mode)
        df_do.set_index(df_do.index+mode)
        
        acc_up = pd.concat([acc_up,df_up])
        acc_do = pd.concat([acc_do,df_do])
        
    acc_up = acc_up.to_dict()
    acc_do = acc_do.to_dict()
    
    acc_unc[syst] = {}
    
    for k,v in acc_up.items():
        acc_unc[syst][k[0]] = {}
        for j,u in v.items():
            acc_unc[syst][k[0]][j] = (acc_up[k][j],acc_do[k][j])

# Main method                                                                                                                           
def main():

    if len(sys.argv) < 2:
        print("Enter year")
        return

    year = sys.argv[1]

    templates_reco = pickle.load(open(year+'/templates.pkl','rb')).sum('msd1')

    templates_true = pickle.load(open(year+'/acceptance-truth.pkl','rb')).sum('pth',overflow='allnan')
    
    true = templates_true
    reco_pass = templates_reco.integrate('ddb1',int_range=slice(0.64,1))
    reco_fail = templates_reco.integrate('ddb1',int_range=slice(0,0.64))

    acc_unc_pass = {}    
    acceptance_uncertainty(acc_unc_pass,reco_pass,true,'scalevar_7pt')
    acceptance_uncertainty(acc_unc_pass,reco_pass,true,'scalevar_3pt')
    acceptance_uncertainty(acc_unc_pass,reco_pass,true,'PDFaS_weight')
    acceptance_uncertainty(acc_unc_pass,reco_pass,true,'UEPS_FSR')
    acceptance_uncertainty(acc_unc_pass,reco_pass,true,'UEPS_ISR')
            
    acc_unc_fail = {}
    acceptance_uncertainty(acc_unc_fail,reco_fail,true,'scalevar_7pt')
    acceptance_uncertainty(acc_unc_fail,reco_fail,true,'scalevar_3pt')
    acceptance_uncertainty(acc_unc_fail,reco_fail,true,'PDFaS_weight')
    acceptance_uncertainty(acc_unc_fail,reco_fail,true,'UEPS_FSR')
    acceptance_uncertainty(acc_unc_fail,reco_fail,true,'UEPS_ISR')    

    acc_unc = {'fail':acc_unc_fail,'pass':acc_unc_pass}

    with open(year+"/acceptance.json", "w") as outfile:
        json.dump(acc_unc, outfile)
        
if __name__ == "__main__":
    main()
