import os
import numpy as np
import json

if __name__ == '__main__':

    year = "2016"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    elif "2017" in thisdir:
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    for pt in range(0,1):
        for rho in range(0,4):
            print("pt = "+str(pt)+", rho = "+str(rho))
            # Make the directory and go there
            thedir = "pt"+str(pt)+"rho"+str(rho)
            if not os.path.isdir(thedir):
                os.mkdir(thedir)
                os.chdir(thedir)

            linkdir='/uscms/home/jennetd/nobackup/hbb-prod-modes/CMSSW_10_2_13/src/vbf-ul/main-vbf/unblinding/'

            # Link what you need
            os.system("mkdir plots")
            os.system("ln -s "+linkdir+year+"/signalregion.root .")
            os.system("ln -s "+linkdir+year+"/muonCR.root .")
            os.system("cp "+linkdir+year+"/initial_vals_ggf.json .")
            os.system("cp "+linkdir+year+"/initial_vals_vbf*.json .")
            os.system("ln -s "+linkdir+"../lumi.json .")
            os.system("ln -s "+linkdir+"sf.json .")
            os.system("ln -s ../../make_cards.py .")
            os.system("ln -s ../../make_workspace.sh .")

            # Create your json files of initial values
#            if not os.path.isfile("initial_vals_data_ggf.json"):

#                initial_vals = (np.full((pt+1,rho+1),1)).tolist()
#                thedict = {}
#                thedict["initial_vals"] = initial_vals
                        
#                with open("initial_vals_data_ggf.json", "w") as outfile:
#                    json.dump(thedict,outfile)

            # Create your json files of initial values                                                                           
            if not os.path.isfile("initial_vals_data_vbf.json"):
                initial_vals = (np.full((pt+1,rho+1),1)).tolist()
                thedict = {}
                thedict["initial_vals"] = initial_vals

                with open("initial_vals_data_vbf.json", "w") as outfile:
                    json.dump(thedict,outfile)

            # Create your json files of initial values                                                                                          
#            if not os.path.isfile("initial_vals_data_vbfhi.json"):
#                initial_vals = (np.full((pt+1,rho+1),1)).tolist()
#                thedict = {}
#                thedict["initial_vals"] = initial_vals

#                with open("initial_vals_data_vbfhi.json", "w") as outfile:
#                    json.dump(thedict,outfile)

            # Make the workspace
            os.system("./make_workspace.sh")

            # Run the first fit
            combine_cmd = "combineTool.py -M MultiDimFit -m 125 -d output/testModel_"+year+"/model_combined.root --saveWorkspace \
            -P rVBF --floatOtherPOIs=0 -n \"Snapshot\" \
            --robustFit=1 --cminDefaultMinimizerStrategy=0 \
            --setParameters rVBF=1"
            os.system(combine_cmd)

            # Run the goodness of fit
            combine_cmd = "combineTool.py -M GoodnessOfFit -m 125 -d higgsCombineSnapshot.MultiDimFit.mH125.root \
            -n \"Observed\" --snapshotName MultiDimFit --bypassFrequentistFit --algo \"saturated\" \
            --setParameters rVBF=1"
            os.system(combine_cmd)
                
            os.chdir("../")
