import os
import math
import numpy as np
import ROOT
import matplotlib.pyplot as plt
from scipy.stats import f
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

if __name__ == '__main__':

    year = "All years"
    thisdir = os.getcwd()
    if "2016APV" in thisdir:
        year = "2016APV"
    elif "2016" in thisdir:
        year = "2016"
    elif "2017" in thisdir:
        year = "2017"
    elif "2018" in thisdir:
        year = "2018"

    gof_toys = []

    infile1 = ROOT.TFile.Open("higgsCombineToys.GoodnessOfFit.mH125.root")
    tree1= infile1.Get("limit")
    for j in range(tree1.GetEntries()):
        tree1.GetEntry(j)
        gof_toys += [getattr(tree1,"limit")]

    ntoys = tree1.GetEntries()

    # Observed
    infile1 = ROOT.TFile.Open("higgsCombineObserved.GoodnessOfFit.mH125.root")
    tree1= infile1.Get("limit")
    tree1.GetEntry(0)
    gof_obs = getattr(tree1,"limit")

    ashist = plt.hist(gof_toys,bins=20,histtype='step',color='black')
    ymax = 1.2*max(ashist[0])

    plt.errorbar((ashist[1][:-1]+ashist[1][1:])/2., ashist[0], yerr=np.sqrt(ashist[0]),linestyle='',color='black',marker='o',label=str(ntoys) +" toys")
    mylabel = "obs = {:.2f}".format(gof_obs)

    pvalue = 1.0*len([y for y in gof_toys if y>gof_obs])/ntoys
    mylabel += ", p = {:.2f}".format(pvalue)
    print(pvalue)

    plt.ylim(0,ymax)
    plt.plot([gof_obs,gof_obs],[0,ymax],color='red',label=mylabel)
    plt.legend(loc='upper right',frameon=False,title=year)
    plt.xlabel("Goodness of fit (saturated)")

#    plt.savefig(thisdir+"/plots/gof.png",bbox_inches='tight')
#    plt.savefig(thisdir+"/plots/gof.pdf",bbox_inches='tight')
    plt.show()
