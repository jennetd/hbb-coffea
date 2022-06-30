#!/bin/bash
echo "Starting job on " `date` #Date/time of start of job                                                                  
echo "Running on: `uname -a`" #Condor job is running on this node                                                           
echo "System software: `cat /etc/redhat-release`" #Operating System on that node    

# CMSSW                                                                                                                    
source /cvmfs/cms.cern.ch/cmsset_default.sh
tar -xzvf CMSSW_10_2_13.tar.gz
cd CMSSW_10_2_13/src/

eval `scramv1 runtime -sh`
 
#Move other stuff in
mv ../../tar_ball.tar.gz .
mv ../../PRE_SCAN_FILE .
mv ../../MAIN_SCAN_FILE .

#Untar the package
tar -xzvf tar_ball.tar.gz

#Run pre-scan
chmod +x PRE_SCAN_FILE
singularity exec -B ${PWD}:/srv --pwd /srv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest /srv/./PRE_SCAN_FILE

echo "HERERERERERER"
pwd
ls -alrth

#Run main scan
chmod +x MAIN_SCAN_FILE
./MAIN_SCAN_FILE
