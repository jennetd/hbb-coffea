#!/bin/bash                                                                                                                                                                                   
echo "Starting job on " `date` #Date/time of start of job                                                                                                                                      
echo "Running on: `uname -a`" #Condor job is running on this node                                                                                                                              
echo "System software: `cat /etc/redhat-release`" #Operating System on that node                                                                                                               
# bring in the tarball you created before with caches and large files excluded:                                                                                                                  
xrdcp -s root://cmseos.fnal.gov//store/user/jennetd/CMSSW_10_2_13.tar.gz .
source /cvmfs/cms.cern.ch/cmsset_default.sh
tar -xf CMSSW_10_2_13.tar.gz
rm CMSSW_10_2_13.tar.gz
cd CMSSW_10_2_13/src/
scramv1 b ProjectRename # this handles linking the already compiled code - do NOT recompile                                                                                                      
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers                                                                                                                               
echo $CMSSW_BASE "is the CMSSW we have on the local worker node"
cd ${_CONDOR_SCRATCH_DIR}
pwd

ls

# Arguments                                                                                                                                                                                    
frozen="allConstrainedNuisances"

combine -M MultiDimFit --algo singles -d higgsCombine.Test.MultiDimFit.mH125.root -m 125 --robustFit=1 --saveWorkspace --setParameters rVBF1=4.9,rVBF7=1,rVBF8=1,rggF1=1,rggF2=1,rggF3=1,rggF4=1,rggF5=1,rggF6=1,rggF7=1.6 --freezeParameters rVBF1,rggF7,${frozen} -n 'Stat' -w w --snapshotName "MultiDimFit"
