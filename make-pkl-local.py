#!/usr/bin/python  

import os, sys, re
import subprocess
import json
import uproot3
import awkward as ak
import numpy as np
from coffea import processor, util, hist
import pickle

lumis = {}
lumis['2016'] = 35.9
lumis['2017'] = 41.5
lumis['2018'] = 59.2

coffeadir_prefix = '/myeosdir/ggf-vbf-plots/outfiles-ddb2/'

# Main method
def main():

    raw = False

    if len(sys.argv) < 3:
        print("Enter year and name")
        return 

    year = sys.argv[1]
    name = sys.argv[2]

    with open('xsec.json') as f:
        xs = json.load(f)
        
    with open('pmap.json') as f:
        pmap = json.load(f)
            
    indir = "infiles-split/"
    nfiles = len(subprocess.getoutput("ls "+indir+year+"*.json").split())
    outsum = processor.dict_accumulator()

    # Check if pickle exists, remove it if it does
    picklename = str(year)+'/'+name+'.pkl'
    if os.path.isfile(picklename):
        os.remove(picklename)

    nfiles = len(subprocess.getoutput("ls "+indir+year+"*.json").split())

    started = 0
    for n in range(1,nfiles+1):

#        print(n)
        with open('infiles-split/'+year+'_'+str(n)+'.json') as f:
            infiles = json.load(f)

        filename = coffeadir_prefix+'/'+year+'/'+year+'_'+str(n)+'.coffea'
        if os.path.isfile(filename):
            out = util.load(filename)

            if started == 0:
                outsum[name] = out[name]
                outsum['sumw'] = out['sumw']
                started += 1
            else:
                outsum[name].add(out[name])
                outsum['sumw'].add(out['sumw'])

            del out

        else:
            print('Missing file '+str(n),infiles.keys())
            #print("File " + filename + " is missing")
        
    scale_lumi = {k: xs[k] * 1000 *lumis[year] / w for k, w in outsum['sumw'].items()} 

    outsum[name].scale(scale_lumi, 'dataset')
    templates = outsum[name].group('dataset', hist.Cat('process', 'Process'), pmap)

    del outsum
          
    outfile = open(picklename, 'wb')
    pickle.dump(templates, outfile, protocol=-1)
    outfile.close()

    return

if __name__ == "__main__":

    main()
