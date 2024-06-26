#!/bin/bash                                                                     
export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates/

echo "Starting job on " `date` #Date/time of start of job                       
echo "System software: `cat /etc/redhat-release`" #Operating System on that nod\
e                                                                               

# bring in the tarball you created before with caches and large files excluded:
xrdcp -s root://cmseos.fnal.gov//store/user/jennetd/CMSSW_10_2_13.tar.gz .
tar -xf CMSSW_10_2_13.tar.gz
rm CMSSW_10_2_13.tar.gz
cd CMSSW_10_2_13/src/
source /cvmfs/cms.cern.ch/cmsset_default.sh
scram b ProjectRename # this handles linking the already compiled code - do NOT recompile                                                                      
eval `scram runtime -sh` # cmsenv is an alias not on the workers                
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ${_CONDOR_SCRATCH_DIR}
pwd


# Arguments
year=$1

modelfile=output/testModel${year}/model_combined.root

# Do initial fit                                                                                          
combine -M MultiDimFit --algo singles -d $modelfile -m 125 --setParameters rggF=1,rVBF=1 --robustFit=1 --robustHesse=1 --cminDefaultMinimizerStrategy=0 --saveWorkspace --verbose 9


