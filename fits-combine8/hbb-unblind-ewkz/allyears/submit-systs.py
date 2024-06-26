#!/usr/bin/python
import os, sys
import subprocess
import argparse

# Main method                                                                          
def main():

    year = "prefit"
    yearstr = ""
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV_prefit"
        yearstr = "_2016APV"
    elif "2016" in thisdir:
        year = "2016-prefit"
        yearstr = "_2016"
    elif "2017" in thisdir:
        year = "2017-prefit"
        yearstr = "_2017"
    elif "2018" in thisdir:
        year = "2018-prefit"
        yearstr = "_2018"

    print("Submitting for year " + year)

    loc_base = os.environ['PWD']
    logdir = 'logs'
    locdir = logdir
    os.system('mkdir -p  %s' %locdir)

    outdir = '/store/user/jennetd/fits/'+year+'/'
    os.system('mkdir -p /eos/uscms'+outdir)

    frozen = {}

    frozen["DDB"] ="rgx{CMS_hbb_eff.*}"
    frozen["JESJER"]="rgx{CMS_res_j.*},rgx{CMS_scale_j.*},rgx{CMS_ues_j.*}"
    frozen["JMSJMR"]="rgx{CMS_hbb_scale.*},rgx{CMS_hbb_smear.*}"
    frozen["SigExtr"]="rgx{CMS_.*},rgx{QCDscale_.*},rgx{UEPS_.*},rgx{pdf_.*},rgx{tf.*},rgx{qcd.*},rgx{ttq.*},rgx{.*mcstat}"
    frozen["VBFtheory"] = "QCDScale_VBF,pdf_VBF,UEPS_ISR_VBF,UEPS_FSR_VBF"
    frozen["VHttHtheory"]="QCDScale_VH,pdf_VH,UEPS_ISR_VH,UEPS_FSR_VH,QCDScale_ttH,pdf_ttH,UEPS_ISR_ttH,UEPS_FSR_ttH"
    frozen["Vtheory"]="rgx{CMS_hbb_.*NLO},rgx{CMS_hbb_.*EW}"
    frozen["ggFtheory"]="QCDScale_ggF,pdf_ggF,UEPS_ISR_ggF,UEPS_FSR_ggF"
    frozen["MCStat"]="rgx{.*mcstat}"
    frozen["Other"]="rgx{CMS_L1Prefiring.*},rgx{CMS_hbb_PU.*},rgx{CMS_.*trigger.*},rgx{CMS_.*mu.*},rgx{CMS_hbb_e_veto.*},rgx{CMS_hbb_tau_veto.*},rgx{CMS_hbb_btagWeight.*},rgx{CMS_hbb_vbfmucr.*},rgx{CMS_hbb_veff.*},rgx{CMS_lumi_13TeV.*}"
    frozen["qcdparam"]="rgx{qcdparam.*}"


    frozen["TFres"] ="rgx{tf_dataResidual.*}"
    frozen["TFMC"] = "rgx{tf_MC.*}"
    frozen["TTQ"] = "rgx{ttq.*}"

    frozen["AllStat"]="rgx{CMS_.*},rgx{QCDscale_.*},rgx{UEPS_.*},rgx{pdf_.*},rgx{qcd.*},rgx{.*mcstat}"
    frozen["AllExp"]="rgx{CMS_hbb_eff.*},rgx{CMS_res_j.*},rgx{CMS_scale_j.*},rgx{CMS_ues_j.*},rgx{CMS_hbb_scale.*},rgx{CMS_hbb_smear.*},rgx{.*mcstat},rgx{CMS_L1Prefiring.*},rgx{CMS_hbb_PU.*},rgx{CMS_.*trigger.*},rgx{CMS_.*mu.*},rgx{CMS_hbb_e_veto.*},rgx{CMS_hbb_tau_veto.*},rgx{CMS_hbb_btagWeight.*},rgx{CMS_hbb_vbfmucr.*},rgx{CMS_hbb_veff.*},rgx{CMS_lumi_13TeV.*},rgx{qcdparam.*}"
    frozen["AllTh"]="rgx{QCDScale_.*},rgx{pdf_.*},rgx{UEPS_.*},rgx{CMS_hbb_.*NLO},rgx{CMS_hbb_.*EW}"
    frozen["otherPOI"] = ""


    for p in ["rggF","rVBF"]:

        if "ggF" in p:
            frozen["otherPOI"] = "rVBF"
        else:
            frozen["otherPOI"] = "rggF"

        for k in frozen.keys():
        
            print('Submitting '+k)
        
            condorfilename = "fit.syst.condor"
        
            condor_templ_file = open(condorfilename)
            localcondor = locdir+'/obs_syst_'+p+"_"+k+".condor"
            condor_file = open(localcondor,"w")

            params_to_freeze = frozen[k] 

            for line in condor_templ_file:
                line=line.replace('FREEZE',params_to_freeze)
                line=line.replace('TAG',k)
                line=line.replace('POI',p)
                condor_file.write(line)
            condor_file.close()
            
            os.system("cp obs_syst.sh "+locdir+"/obs_syst_"+k+".sh")

            if (os.path.exists('%s.log'  % localcondor)):
                os.system('rm %s.log' % localcondor)
            os.system('condor_submit %s' % localcondor)

    return 

if __name__ == "__main__":
    main()
