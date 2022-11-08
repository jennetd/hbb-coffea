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

    if os.path.isfile(year+'/2mjj-signalregion.root'):
        os.remove(year+'/2mjj-signalregion.root')
    fout = uproot3.create(year+'/2mjj-signalregion.root')

    samples = ['data','muondata','QCD','ttbar','singlet','VV','EWKW','EWKZ','ggF','VBF','WH','ZH','ttH']

    print("2 MJJ BINS SR")
    mjjbins = [1000,2000,13000]

    # Check if pickle exists     
    picklename = year+'/templates.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file
    vbf = pickle.load(open(picklename,'rb')).integrate('region','signal-vbf')

    for i,b in enumerate(mjjbins[:-1]):

        for p in samples:
            print(p)

            hpass = vbf.sum('pt1','genflavor').integrate('mjj',int_range=slice(mjjbins[i],mjjbins[i+1])).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail = vbf.sum('pt1','genflavor').integrate('mjj',int_range=slice(mjjbins[i],mjjbins[i+1])).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            for s in hfail.identifiers('systematic'):

                fout["vbf_pass_mjj"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hpass.integrate('systematic',s))
                fout["vbf_fail_mjj"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hfail.integrate('systematic',s))

        for p in ['Wjets','Zjets']:
            print(p)

            hpass = vbf.sum('pt1').integrate('mjj',int_range=slice(mjjbins[i],mjjbins[i+1])).integrate('genflavor',int_range=slice(1,3)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail = vbf.sum('pt1').integrate('mjj',int_range=slice(mjjbins[i],mjjbins[i+1])).integrate('genflavor',int_range=slice(1,3)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            hpass_bb = vbf.sum('pt1').integrate('mjj',int_range=slice(mjjbins[i],mjjbins[i+1])).integrate('genflavor',int_range=slice(3,4)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail_bb = vbf.sum('pt1').integrate('mjj',int_range=slice(mjjbins[i],mjjbins[i+1])).integrate('genflavor',int_range=slice(3,4)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            for s in hfail.identifiers('systematic'):

                fout["vbf_pass_mjj"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hpass.integrate('systematic',s))
                fout["vbf_fail_mjj"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hfail.integrate('systematic',s))

                fout["vbf_pass_mjj"+str(i+1)+"_"+p+"bb_"+str(s)] = hist.export1d(hpass_bb.integrate('systematic',s))
                fout["vbf_fail_mjj"+str(i+1)+"_"+p+"bb_"+str(s)] = hist.export1d(hfail_bb.integrate('systematic',s))

    return

if __name__ == "__main__":
    main()
