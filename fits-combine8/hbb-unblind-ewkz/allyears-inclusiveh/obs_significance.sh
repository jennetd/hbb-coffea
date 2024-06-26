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

echo "INCLUSIVE HBB SIGNIFICANCE"
combine -M Significance -m 125 --signif output/testModel/model_combined.root  --verbose 9 --setParameters rHbb=1
