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
    templates = pickle.load(open(picklename,'rb')).integrate('genflavor',int_range=slice(1,4))

    combine_bins = {}
    combine_bins['ggF'] = {}
    combine_bins['ggF'][0] = [0,1,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25,27]
    combine_bins['ggF'][1] = [2,6]
    combine_bins['ggF'][2] = [3,7]
    combine_bins['ggF'][3] = [4,8] 

    combine_bins['VBF'] = {}
    combine_bins['VBF'][0] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,25,27] 
    combine_bins['VBF'][1] = [21,22]
    combine_bins['VBF'][2] = [23,24]

    # loop over reco bins
    for c in ['ggf-pt1','ggf-pt2','ggf-pt3','ggf-pt4','ggf-pt5','ggf-pt6','vbf-mjj1','vbf-mjj2']:

        name_pass = c[0:3]+"_pass_"+c[-3:]
        name_fail = c[0:3]+"_fail_"+c[-3:]

        if 'vbf' in c:
            name_pass = c[0:3]+"_pass_"+c[-4:]
            name_fail = c[0:3]+"_fail_"+c[-4:]


        thisbin = templates.integrate('category',c)

        # Loop over STXS bins
        for m in ['ggF','VBF']:

            hpass = {}
            hfail = {}

            # Dictionary of hists in terms of fine STXS bin
            for b in thisbin.identifiers('stxs'):
                
                intb = int(str(b).split(',')[0].split('[')[1])

                hpass[intb] = thisbin.integrate('process',m).integrate('stxs',b).sum('pth',overflow='all').integrate('ddb1',int_range=slice(ddbthr,1))
                hfail[intb] = thisbin.integrate('process',m).integrate('stxs',b).sum('pth',overflow='all').integrate('ddb1',int_range=slice(0,ddbthr))

            for b,bins in combine_bins[m].items():
                print('reduced bin',bins)

                p = m + "_s"+str(b)

                hp = hpass[bins[0]]
                hf = hfail[bins[0]]
                for i in bins[1:]:
                    hp = hp + hpass[i]
                    hf = hf + hfail[i]

                # Loop over systematics
                for s in hf.identifiers('systematic'):

                    print(name_pass + "_" + p + "_" + str(s))

                    fout[name_pass + "_" + p + "_" + str(s)] = hist.export1d(hp.integrate('systematic',s))
                    fout[name_fail + "_" + p + "_" + str(s)] = hist.export1d(hf.integrate('systematic',s))

    fout.close()
    return

if __name__ == "__main__":
    main()
