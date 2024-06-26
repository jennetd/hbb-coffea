from ROOT import *

filename = "2016APV/stxs-signalregion.root"
infile = TFile(filename)

for c in ['pt1','pt2','pt3','pt4','pt5','pt6','mjj1','mjj2']:

    print(c)

    mode = 'ggf'
    if 'mjj' in c:
        mode = 'vbf' 

    for j in range(0,17):
        h = infile.Get(mode+'_pass_'+c+'_ggF_s'+str(j)+'_nominal')
        content = h.Integral()
        if content > 0:
            print('ggF',j,content)

        h = infile.Get(mode+'_pass_'+c+'_VBF_s'+str(j)+'_nominal')
        content = h.Integral()
        if content > 0:
            print('VBF',j,content)
