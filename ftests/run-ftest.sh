#!/bin/bash
export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates/

echo "Starting job on " `date` #Date/time of start of job               
echo "System software: `cat /etc/redhat-release`" #Operating System on that node                           

# bring in the tarball you created before with caches and large files excluded:
xrdcp -s root://cmseos.fnal.gov//store/user/jennetd/CMSSW_11_3_4.tar.gz .
tar -xf CMSSW_11_3_4.tar.gz
rm CMSSW_11_3_4.tar.gz
cd CMSSW_11_3_4/src/
source /cvmfs/cms.cern.ch/cmsset_default.sh
scram b ProjectRename # this handles linking the already compiled code - do NOT recompile                         
eval `scram runtime -sh` # cmsenv is an alias not on the workers     
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ${_CONDOR_SCRATCH_DIR}
pwd

# My job
echo "Arguments passed to the job: "
echo $1
echo $2
echo $3
echo $4

eosout=$3
index=$4

python compare.py --pt=$1 --rho=$2 --ntoys=50 --index=$4

dirs=`ls | grep pt$1rho$2_vs_`
for d in $dirs;
do
    #move output to eos
    xrdfs root://cmseos.fnal.gov/ mkdir $eosout/${d}_$index
    xrdcp -rf $d root://cmseos.fnal.gov/$eosout/${d}_$index
done
