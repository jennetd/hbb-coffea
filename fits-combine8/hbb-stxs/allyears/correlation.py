import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(70)

import numpy as np
import matplotlib.pyplot as plt

import json

filename = "robustHesseTest.root"
f = ROOT.TFile(filename)

h = f.Get('h_correlation')

corr = ROOT.TH2F("corr","corr",5,0.5,5.5,5,0.5,5.5)

pycorr = []
for i in range(1,h.GetNbinsX()+1):
    xlabel = h.GetXaxis().GetBinLabel(i)
    
    if 'rVBF' not in xlabel and 'rggF' not in xlabel:
        continue
    for j in range(1,h.GetNbinsY()+1):
        ylabel = h.GetYaxis().GetBinLabel(j)
        
        if 'rVBF' not in ylabel and 'rggF' not in ylabel:
            continue
        
        #print(xlabel,ylabel)
        corr.Fill(xlabel,ylabel,h.GetBinContent(i,j))


c = ROOT.TCanvas()
c.Draw()
corr.Draw("COLZtext")
corr.GetZaxis().SetRangeUser(-1,1)
c.Update()
pycorr = np.array(corr).reshape(7,7)[1:6,1:6]
plt.imshow(pycorr)

np.save("pycorr.npy", pycorr)
