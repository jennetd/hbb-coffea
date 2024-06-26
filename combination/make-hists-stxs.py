#!/usr/bin/python  

import os, sys
import subprocess
import json
import uproot3
import awkward as ak
import numpy as np
from coffea import processor, util, hist
import pickle

with open('lumi.json') as f:
    lumis = json.load(f)

ddbthr = 0.64

# Main method
def main():

    if len(sys.argv) < 2:
        print("Enter year")
        return
    elif len(sys.argv) > 3:
        print("Incorrect number of arguments")
        return

    year = sys.argv[1]

    if os.path.isfile(year+'/stxs-signalregion.root'):
        os.remove(year+'/stxs-signalregion.root')
    fout = uproot3.create(year+'/stxs-signalregion.root')

    # Check if pickle exists     
    picklename = year+'/templates.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file                                                                      
    templates = pickle.load(open(picklename,'rb'))

    # loop over reco bins
    for c in ['ggf-pt1','ggf-pt2','ggf-pt3','ggf-pt4','ggf-pt5','ggf-pt6','vbf-mjj1','vbf-mjj2']:

        name_pass = c[0:3]+"_pass_"+c[-3:]
        name_fail = c[0:3]+"_fail_"+c[-3:]

        if 'vbf' in c:
            name_pass = c[0:3]+"_pass_"+c[-4:]
            name_fail = c[0:3]+"_fail_"+c[-4:]


        thisbin = templates.integrate('category',c)

        print(thisbin.identifiers('mode'))
        
        # Loop over STXS bins
        for m in range(1,9):

            hpass = {}
            hfail = {}

            # Dictionary of hists in terms of fine STXS bin
            for b in thisbin.identifiers('stxs'):
                
                intb = int(str(b).split(',')[0].split('[')[1])

                hpass[intb] = thisbin.integrate('mode',int_range=slice(m,m+1)).integrate('stxs',b).integrate('ddb1',int_range=slice(ddbthr,1))
                hfail[intb] = thisbin.integrate('mode',int_range=slice(m,m+1)).integrate('stxs',b).integrate('ddb1',int_range=slice(0,ddbthr))

                hp = hpass[intb]
                hf = hfail[intb]

                p = "s"+str(m*100+intb)

                # Loop over systematics
                for s in hf.identifiers('systematic'):

                    print(name_pass + "_" + p + "_" + str(s))

                    fout[name_pass + "_" + p + "_" + str(s)] = hist.export1d(hp.integrate('systematic',s))
                    fout[name_fail + "_" + p + "_" + str(s)] = hist.export1d(hf.integrate('systematic',s))

    fout.close()
    return

if __name__ == "__main__":
    main()
