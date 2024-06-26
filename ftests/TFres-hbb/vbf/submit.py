#!/usr/bin/python
import os, sys
import subprocess
import argparse

# Main method                                                                          
def main():

    year = "2016"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    elif "2017" in thisdir:
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    parser = argparse.ArgumentParser(description='F-test batch submit')
    parser.add_argument('-p','--pt',nargs='+',help='pt of baseline')
    parser.add_argument('-r','--rho',nargs='+',help='rho of baseline')
    parser.add_argument('-n','--njobs',nargs='+',help='number of 50 toy jobs to submit')
    args = parser.parse_args()

    pt = int(args.pt[0])
    rho = int(args.rho[0])
    njobs = int(args.njobs[0])

    loc_base = os.environ['PWD']
    logdir = 'logs'

    tag = year+"_vbf_pt" + str(pt) + "rho" + str(rho)
    script = 'run-ftest.sh'

    homedir = '/store/user/jennetd/hbb-f-tests-unblinded/'
    outdir = homedir + tag 

    # make local directory
    locdir = logdir
    os.system('mkdir -p  %s' %locdir)

    print('CONDOR work dir: ' + homedir)
    os.system('mkdir -p /eos/uscms'+outdir)

    transferfiles = "compare.py,pt" + str(pt) + "rho" + str(rho)

    alternatives = []
#    alternatives += ["pt"+str(pt+1)+"rho"+str(rho)]
    alternatives += ["pt"+str(pt)+"rho"+str(rho+1)]

    alternatives = list(set(alternatives))
    for a in alternatives:
        transferfiles += ","+a

    for i in range(0,njobs):
        prefix = tag+"_"+str(i)
        print('Submitting '+prefix)

        condor_templ_file = open("submit.templ.condor")

        submitargs = str(pt) + " " + str(rho) + " " + outdir + " " + str(i)
    
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
