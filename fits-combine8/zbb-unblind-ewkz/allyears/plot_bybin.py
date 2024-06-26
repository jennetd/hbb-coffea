import ROOT as rt
from array import array
from scipy.interpolate import Rbf
import numpy as np
import glob
import pandas as pd
import argparse

def read_from_file(poi, filestring):

    print(poi)
    search_string = "   r"+poi+" :    "

    vals = []

    theline = ""

    with open(filestring, "r+") as readfile:
        for line in readfile.readlines():
            if search_string in line:
                theline = line

    chunks = theline.split()
    vals += [float(chunks[2])]
    uncs = chunks[3].split('/')
    vals += [abs(float(uncs[0]))]
    vals += [float(uncs[1])]

    return vals

def get_total(poi):

    total = read_from_file(poi,"../allyears/logs/fit_batch.out")
    print(total)

    n = 2
    center = np.array([total[0],total[0]])
    up_unc = np.array([total[2],total[2]])
    do_unc = np.array([total[1],total[1]])
    x = np.array([-100.,100.])

    return rt.TGraphAsymmErrors(n,center,x,do_unc,up_unc,np.zeros(2),np.zeros(2))

def get_total_uncertainty(poi, logfile):
    total = read_from_file(poi,logfile)
    print(total)
  
    n = 2
    x = np.array([-100.,100.])

    return rt.TGraphAsymmErrors(n,np.array([total[0]-total[1],total[0]+total[2]]),np.zeros(2), np.zeros(2),np.zeros(2), x,x)

def get_Zbb_table(zbb_logfile):

    bin1 = read_from_file("Zbb1",zbb_logfile)
    bin2 = read_from_file("Zbb2",zbb_logfile)
    bin3 = read_from_file("Zbb3",zbb_logfile)
    bin4 = read_from_file("Zbb4",zbb_logfile)
    bin5 = read_from_file("Zbb5",zbb_logfile)
    bin6 = read_from_file("Zbb6",zbb_logfile)
    bin7 = read_from_file("Zbb7",zbb_logfile)
    bin8 = read_from_file("VBF8",zbb_logfile)

    n = 8

    center = np.array([bin1[0],bin2[0],bin3[0],bin4[0],bin5[0],bin6[0],bin7[0],bin8[0]])
    up_unc = np.array([bin1[2],bin2[2],bin3[2],bin4[2],bin5[2],bin6[2],bin7[2],bin8[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1],bin4[1],bin5[1],bin6[1],bin7[1],bin8[1]])
    x = np.array([1.,2.,3.,4.,5.,6.,7.,8.])
    
    return rt.TGraphAsymmErrors(n,center,x,do_unc,up_unc,np.zeros(2),np.zeros(2))

rt.gStyle.SetOptTitle(0)
rt.gStyle.SetOptStat(0)
rt.gStyle.SetEndErrorSize(0)

c = rt.TCanvas("c", "c", 800, 800)
textsize1 = 0.04

rt.gPad.SetLeftMargin(0.3)
rt.gPad.SetBottomMargin(0.15)

lumi = 138
tag1 = rt.TLatex(0.63, 0.92, "%.0f fb^{-1} (13 TeV)" % lumi)
tag1.SetNDC()
tag1.SetTextFont(42)
tag1.SetTextSize(textsize1)
tag2 = rt.TLatex(0.32, 0.92, "#bf{CMS}")
tag2.SetNDC()
tag2.SetTextFont(42)
tag2.SetTextSize(textsize1)

n = 2
smy = np.array([-100.,100.])
smx = np.array([1.,1.])

sm = rt.TGraph(n,smx,smy)
sm.SetLineColor(1);
sm.SetLineStyle(3);
sm.SetLineWidth(3);

smy = np.array([-100.,100.])
zerox = np.array([0.,0.])

zero = rt.TGraph(n,zerox,smy)
zero.SetLineColor(0);
zero.SetLineWidth(3);

h1 = rt.TH1D("dummy1","dummy1",8,0.5,8.5)
h1.Fill("1000 < m_{jj} < 2000 GeV",-15)
h1.Fill("m_{jj} > 2000 GeV",-15)
h1.Fill("450 < p_{T} < 500 GeV",-15)
h1.Fill("500 < p_{T} < 550 GeV",-15)
h1.Fill("550 < p_{T} < 600 GeV",-15)
h1.Fill("600 < p_{T} < 675 GeV",-15)
h1.Fill("675 < p_{T} < 800 GeV",-15)
h1.Fill("800 < p_{T} < 1200 GeV",-15)

h1.SetLineColor(0)
#h1.GetXaxis().SetTitle("Bin")
h1.GetXaxis().SetTitleSize(textsize1)
#h1.GetXaxis().SetTitleOffset(1)
h1.GetXaxis().SetLabelSize(textsize1)
h1.GetYaxis().SetTitle('#mu_{Zbb}')
h1.GetYaxis().SetTitleSize(textsize1)
h1.GetYaxis().SetTitleOffset(1.5)
h1.GetYaxis().SetLabelSize(textsize1)
h1.GetYaxis().SetRangeUser(0,1.2)
h1.Draw("hbar")

sm.Draw("lsame")
#    zero.Draw("lsame")

gzunc = get_total_uncertainty("Zbb","logs/fit_batch.out")
gzunc.SetLineColor(4)
gzunc.SetFillColor(4)
gzunc.SetFillStyle(3003)
gzunc.SetLineWidth(3)
gzunc.Draw("3same")

gz = get_total("Zbb")
gz.SetLineColor(4)
gz.SetFillColor(4)
gz.SetFillStyle(3003)
gz.SetLineWidth(3)
gz.Draw("lsame")

gz2 = get_Zbb_table("../allyears-bybin/logs/fit_batch.out")
gz2.SetMarkerColor(1)
gz2.SetMarkerStyle(20)
gz2.SetLineColor(1)
gz2.SetLineWidth(3)
gz2.Draw("psame")

tag1.Draw()
tag2.Draw()
#    tag3.Draw()

leg = rt.TLegend(0.35, 0.3, 0.5, 0.45)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetFillColor(rt.kWhite)
leg.SetLineColor(rt.kWhite)
leg.SetLineStyle(0)
leg.SetFillStyle(0)
leg.SetLineWidth(0)

leg.AddEntry(gz2,"Observed","p")
leg.AddEntry(gz2,"#pm1#sigma (stat #oplus syst)","l")
#leg.AddEntry(gggf4,"#pm1#sigma (stat)","f")
leg.AddEntry(gz,"Combined fit","lf")
#leg.AddEntry(sm,"SM expectation","l")

leg.Draw("same")

c.Print("mu_bybin.pdf")
c.Print("mu_bybin.png")
c.Print("mu_bybin.C")

c.Draw()
