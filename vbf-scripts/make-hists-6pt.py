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

    if raw:
        year = year+"-raw"
    if os.path.isfile(year+'/6pt-signalregion.root'):
        os.remove(year+'/6pt-signalregion.root')
    fout = uproot3.create(year+'/6pt-signalregion.root')

    samples = ['data','muondata','QCD','ttbar','singlet','VV','ggF','VBF','WH','ZH','ttH']

    print("6 PT BINS ggF SR")
    ptbins = [450, 500, 550, 600, 675, 800, 1200]

    # Check if pickle exists     
    picklename = year+'/templates.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file                                                                      
    ggf = pickle.load(open(picklename,'rb')).integrate('region','signal-ggf').integrate('mjj',overflow='allnan')

    for i,b in enumerate(ptbins[:-1]):

        for p in samples:
            print(p)

            hpass = ggf.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).sum('genflavor').integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail = ggf.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).sum('genflavor').integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            if year == '2016' and p == 'ggF' and not raw:
                print("Taking shape for 2016 ggF from 2017")
                ggf17 = pickle.load(open('2017/templates.pkl','rb')).integrate('region','signal-ggf').integrate('mjj',overflow='allnan')
                ggf17.scale(lumis['2016']/lumis['2017'])

                hpass = ggf17.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).sum('genflavor').integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
                hfail = ggf17.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).sum('genflavor').integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            for s in hfail.identifiers('systematic'):

                fout["ggf_pass_pt"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hpass.integrate('systematic',s))
                fout["ggf_fail_pt"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hfail.integrate('systematic',s))

        for p in ['Wjets','Zjets']:
            print(p)

            hpass = ggf.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).integrate('genflavor',int_range=slice(1,3)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail = ggf.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).integrate('genflavor',int_range=slice(1,3)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            hpass_bb = ggf.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).integrate('genflavor',int_range=slice(3,4)).integrate('ddb1',int_range=slice(ddbthr,1)).integrate('process',p)
            hfail_bb = ggf.integrate('pt1',int_range=slice(ptbins[i],ptbins[i+1])).integrate('genflavor',int_range=slice(3,4)).integrate('ddb1',int_range=slice(0,ddbthr)).integrate('process',p)

            for s in hfail.identifiers('systematic'):

                fout["ggf_pass_pt"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hpass.integrate('systematic',s))
                fout["ggf_fail_pt"+str(i+1)+"_"+p+"_"+str(s)] = hist.export1d(hfail.integrate('systematic',s))

                fout["ggf_pass_pt"+str(i+1)+"_"+p+"bb_"+str(s)] = hist.export1d(hpass_bb.integrate('systematic',s))
                fout["ggf_fail_pt"+str(i+1)+"_"+p+"bb_"+str(s)] = hist.export1d(hfail_bb.integrate('systematic',s))

    fout.close()

    return

if __name__ == "__main__":
    main()
