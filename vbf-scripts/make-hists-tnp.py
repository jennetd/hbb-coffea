#!/usr/bin/python  

import os, sys
import subprocess
import json
import uproot3
import awkward as ak
import numpy as np
from coffea import processor, util, hist
import pickle

n2ddtthr = 0

# Main method
def main():

    if len(sys.argv) < 2:
        print("Enter year")
        return

    elif len(sys.argv) > 3:
        print("Incorrect number of arguments")
        return

    year = sys.argv[1]

    if os.path.isfile(year+'/TnPtemplates.root'):
        os.remove(year+'/TnPtemplates.root')
    fout = uproot3.create(year+'/TnPtemplates.root')

    # Check if pickle exists     
    picklename = year+'/templates-tnp.pkl'
    if not os.path.isfile(picklename):
        print("You need to create the pickle")
        return

    # Read the histogram from the pickle file
    tnp = pickle.load(open(picklename,'rb')).integrate('region','tnp').sum('ddb1')

    # data first
    p = "muondata"
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).sum('genflavor')
    hfail = tnp.integrate('n2ddt',int_range=slice(0,1)).integrate('process',p).sum('genflavor')

    fout["pass_"+p+"_nominal"] = hist.export1d(hpass)
    fout["fail_"+p+"_nominal"] = hist.export1d(hfail)

    # samples included
    p = ["ttbar","singlet","QCD","Wjets","Zjets"]

    # matched
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(1,4))
    hfail = tnp.integrate('n2ddt',int_range=slice(0,1)).integrate('process',p).integrate('genflavor',int_range=slice(1,4))
    fout["pass_match_nominal"] = hist.export1d(hpass)
    fout["fail_match_nominal"] = hist.export1d(hfail)

    # unmatched
    hpass = tnp.integrate('n2ddt',int_range=slice(-1,0)).integrate('process',p).integrate('genflavor',int_range=slice(0,1))
    hfail = tnp.integrate('n2ddt',int_range=slice(0,1)).integrate('process',p).integrate('genflavor',int_range=slice(0,1))
    fout["pass_un_nominal"] = hist.export1d(hpass)
    fout["fail_un_nominal"] = hist.export1d(hfail)

    return

if __name__ == "__main__":
    main()
