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

    samples = ['data','muondata','QCD','ttbar','singlet','VV','ggF','VBF','WH','ZH','ttH']

    print("MUON CR")
    if os.path.isfile(year+'/muonCR.root'):
        os.remove(year+'/muonCR.root')
    fout = uproot3.create(year+'/muonCR.root')

    # Check if pickle exists                                                                         
    picklename = year+'/templates.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file
    mucr = pickle.load(open(picklename,'rb')).integrate('region','muoncontrol').integrate('mjj',overflow='allnan')

    for p in samples:
        print(p)

        hpass = mucr.sum('pt1','genflavor').integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
        hfail = mucr.sum('pt1','genflavor').integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

        for s in hfail.identifiers('systematic'):

            fout["pass_"+p+"_"+str(s)] = hist.export1d(hpass.integrate('systematic',s))
            fout["fail_"+p+"_"+str(s)] = hist.export1d(hfail.integrate('systematic',s))

    for p in ['Wjets','Zjets','EWKW','EWKZ']:
        print(p)

        hpass = mucr.sum('pt1').integrate('genflavor',int_range=slice(1,3)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
        hfail = mucr.sum('pt1').integrate('genflavor',int_range=slice(1,3)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

        hpass_bb = mucr.sum('pt1').integrate('genflavor',int_range=slice(3,4)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
        hfail_bb = mucr.sum('pt1').integrate('genflavor',int_range=slice(3,4)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)


        for s in hfail.identifiers('systematic'):

            fout["pass_"+p+"_"+str(s)] = hist.export1d(hpass.integrate('systematic',s))
            fout["fail_"+p+"_"+str(s)] = hist.export1d(hfail.integrate('systematic',s))

            fout["pass_"+p+"bb_"+str(s)] = hist.export1d(hpass_bb.integrate('systematic',s))
            fout["fail_"+p+"bb_"+str(s)] = hist.export1d(hfail_bb.integrate('systematic',s))
        
    fout.close()

    return

if __name__ == "__main__":

    main()
