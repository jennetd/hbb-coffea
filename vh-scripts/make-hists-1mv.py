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

ddbthr = 0.84
ddcthr = 0.05

# Main method
def main():

    raw = False

    if len(sys.argv) < 2:
        print("Enter year")
        return
    elif len(sys.argv) == 3:
        if int(sys.argv[2]) > 0:
            raw = True
    elif len(sys.argv) > 3:
        print("Incorrect number of arguments")
        return

    year = sys.argv[1]

    if os.path.isfile(year+'/1mv-signalregion.root'):
        os.remove(year+'/1mv-signalregion.root')
    fout = uproot3.create(year+'/1mv-signalregion.root')

    samples = ['data','muondata','QCD','ttbar','singlet','VV','ggF','VBF','WH','ZH','ttH']

    print("1 V MASS BINS SR")
    bins = [40,201]

    # Check if pickle exists     
    picklename = year+'/templates.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file
    sig = pickle.load(open(picklename,'rb')).integrate('region','signal').sum('ddc2','genflavor2')

    for i,b in enumerate(bins[:-1]):

        for p in samples:
            print(p)

            hpass = sig.sum('genflavor1').integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail = sig.sum('genflavor1').integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            if year == '2016' and p == 'ggF':
                print("Taking shape for 2016 ggF from 2017")
                sig17 = pickle.load(open('2017/templates.pkl','rb')).integrate('region','signal')
                sig17.scale(lumis['2016']/lumis['2017'])

                hpass = sig17.sum('genflavor1').integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
                hfail = sig17.sum('genflavor1').integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            fout["pass_mv"+str(i+1)+"_"+p+"_nominal"] = hist.export1d(hpass)
            fout["fail_mv"+str(i+1)+"_"+p+"_nominal"] = hist.export1d(hfail)

        for p in ['Wjets','Zjets']:
            print(p)

            hpass = sig.integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('genflavor1',int_range=slice(1,3)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail = sig.integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('genflavor1',int_range=slice(1,3)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            hpass_bb = sig.integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('genflavor1',int_range=slice(3,4)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail_bb = sig.integrate('msd2',int_range=slice(bins[i],bins[i+1])).integrate('genflavor1',int_range=slice(3,4)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            fout["pass_mv"+str(i+1)+"_"+p+"_nominal"] = hist.export1d(hpass)
            fout["fail_mv"+str(i+1)+"_"+p+"_nominal"] = hist.export1d(hfail)

            fout["pass_mv"+str(i+1)+"_"+p+"bb_nominal"] = hist.export1d(hpass_bb)
            fout["fail_mv"+str(i+1)+"_"+p+"bb_nominal"] = hist.export1d(hfail_bb)

    return

if __name__ == "__main__":
    main()
