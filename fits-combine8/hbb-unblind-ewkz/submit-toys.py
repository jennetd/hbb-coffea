#!/usr/bin/python
import os, sys
import subprocess
import argparse

# Main method                                                                          
def main():

    year = "allyears"
    modelfile = "output/testModel/model_combined.root"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
        modelfile = "output/testModel_2016APV/model_combined.root"
    elif "2016" in thisdir:
        year = "2016"
        modelfile = "output/testModel_2016/model_combined.root"
    elif "2017" in thisdir:
        year = "2017"
        modelfile = "output/testModel_2017/model_combined.root"
    elif "2018" in thisdir:
        year = "2018"
        modelfile = "output/testModel_2018/model_combined.root"
    
    njobs = 50

#    if year =="allyears":
#        njobs = 200

    loc_base = os.environ['PWD']
    logdir = 'logs-toys'

    tag = year
    script = 'run-toys.sh'

    homedir = '/store/user/jennetd/vbf-toys-unblinding-separated/'
    outdir = homedir + tag 

    # make local directory
    locdir = logdir
    os.system('mkdir -p  %s' %locdir)

    print('CONDOR work dir: ' + homedir)
    os.system('mkdir -p /eos/uscms'+outdir)

    transferfiles = "toys.py,"+modelfile

    for i in range(0,njobs):
        prefix = tag+"_"+str(i)
        print('Submitting '+prefix)

        condor_templ_file = open("toys.templ.condor")

        submitargs = outdir + " " + str(i)
    
        localcondor = locdir+'/'+prefix+".condor"
        condor_file = open(localcondor,"w")
        for line in condor_templ_file:
            line=line.replace('TRANSFERFILES',transferfiles)
            line=line.replace('PREFIX',prefix)
            line=line.replace('SUBMITARGS',submitargs)
            condor_file.write(line)
        condor_file.close()
    
        if (os.path.exists('%s.log'  % localcondor)):
            os.system('rm %s.log' % localcondor)
        os.system('condor_submit %s' % localcondor)

        condor_templ_file.close()
    
    return 

if __name__ == "__main__":
    main()
