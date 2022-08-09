#!/usr/bin/python  

import os, sys
import subprocess
import json
import uproot3
import awkward as ak
import numpy as np
from coffea import processor, util, hist
import pickle

# Main method
def main():

    if len(sys.argv) < 2:
        print("Enter year")
        return 

    year = sys.argv[1]

    with open('xsec.json') as f:
        xs = json.load(f)
        
    with open('pmap.json') as f:
        pmap = json.load(f)

    with open('lumi.json') as f:
        lumis = json.load(f)
            
    indir = "outfiles/"
    infiles = subprocess.getoutput("ls "+indir+year+"_dask_*.coffea").split()
    outsum = processor.dict_accumulator()

    # Check if pickle exists, remove it if it does
    picklename = str(year)+'/templates.pkl'
    if os.path.isfile(picklename):
        os.remove(picklename)

    started = 0
    for filename in infiles:

        print("Loading "+filename)

        if os.path.isfile(filename):
            out = util.load(filename)

            if started == 0:
                outsum['templates'] = out['templates']
                outsum['sumw'] = out['sumw']
                started += 1
            else:
                outsum['templates'].add(out['templates'])
                outsum['sumw'].add(out['sumw'])

            del out

    scale_lumi = {k: xs[k] * 1000 * lumis[year] / w for k, w in outsum['sumw'].items()} 

    outsum['templates'].scale(scale_lumi, 'dataset')

    print(outsum['templates'].identifiers('dataset'))

    templates = outsum['templates'].group('dataset', hist.Cat('process', 'Process'), pmap)

    del outsum
          
    outfile = open(picklename, 'wb')
    pickle.dump(templates, outfile, protocol=-1)
    outfile.close()

    return

if __name__ == "__main__":

    main()
