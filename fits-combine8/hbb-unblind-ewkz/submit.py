#!/usr/bin/python
import os, sys
import subprocess
import argparse

# Main method                                                                          
def main():

    year = "unblinding"
    yearstr = ""
    thisdir = os.getcwd()

    if "2016APV" in thisdir:
        year = "2016APV"
        yearstr = "_2016APV"
    elif "2016" in thisdir:
        year = "2016"
        yearstr = "_2016"
    elif "2017" in thisdir:
        year = "2017"
        yearstr = "_2017"
    elif "2018" in thisdir:
        year = "2018"
        yearstr = "_2018"

    print("Submitting for year " + year)

    loc_base = os.environ['PWD']
    logdir = 'logs'
    locdir = logdir
    os.system('mkdir -p  %s' %locdir)

    outdir = '/store/user/jennetd/fits/'+year+'/'
    os.system('mkdir -p /eos/uscms'+outdir)

    tags = ['fit_batch',
            'obs_shapes',
            'obs_significance'
           ]

    for tag in tags:
        
        print('Submitting '+tag)

        condorfilename = "fit.templ.condor"

        condor_templ_file = open(condorfilename)
        localcondor = locdir+'/'+tag+".condor"
        condor_file = open(localcondor,"w")
        for line in condor_templ_file:
            line=line.replace('YEAR',yearstr)
            line=line.replace('TAG',tag)
            condor_file.write(line)
        condor_file.close()

        if "shapes" in tag or "contour" in tag or "obs_mu" in tag:
            sh_templ_file    = open(tag+".sh")
            localsh = locdir+'/'+tag+'.sh'
            sh_file = open(localsh,"w")
            for line in sh_templ_file:
                line=line.replace('EOSDIR',outdir)
                sh_file.write(line)
            sh_file.close()
        else:
            os.system("cp "+tag+".sh "+locdir)

        if (os.path.exists('%s.log'  % localcondor)):
            os.system('rm %s.log' % localcondor)
        os.system('condor_submit %s' % localcondor)

    return 

if __name__ == "__main__":
    main()
