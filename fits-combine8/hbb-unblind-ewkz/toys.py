import os
import numpy as np
import ROOT
import argparse

def GoF(ntoys, seed=123456):

    infile = "model_combined.root"
    combine_cmd = "combine -M GoodnessOfFit -m 125 -d "+infile+" -n Toys -t " + str(ntoys) + " --algo saturated --seed " + str(seed) + " --toysFrequentist"
    os.system(combine_cmd)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='F-test')
    parser.add_argument('-n','--ntoys',nargs='+',help='number of toys')
    parser.add_argument('-i','--index',nargs='+',help='index for random seed')
    args = parser.parse_args()

    ntoys = int(args.ntoys[0])
    seed = 123456+int(args.index[0])*100+31
    
    # run toys and gof
    GoF(ntoys,seed=seed)
    


