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

def get_VBF_table(vbf_logfile):

    bin7 = read_from_file("VBF7",vbf_logfile)
    bin8 = read_from_file("VBF8",vbf_logfile)

    n = 2
    center = np.array([bin7[0],bin8[0]])
    up_unc = np.array([bin7[2],bin8[2]])
    do_unc = np.array([bin7[1],bin8[1]])
    x = np.array([1.,2.])

    return rt.TGraphAsymmErrors(n,center,x,do_unc,up_unc,np.zeros(2),np.zeros(2))

def get_ggF_table(ggf_logfile):

    bin1 = read_from_file("ggF1",ggf_logfile)
    bin2 = read_from_file("ggF2",ggf_logfile)
    bin3 = read_from_file("ggF3",ggf_logfile)
    bin4 = read_from_file("ggF4",ggf_logfile)
    bin5 = read_from_file("ggF5",ggf_logfile)
    bin6 = read_from_file("ggF6",ggf_logfile)

    n = 6;
    center = np.array([bin1[0],bin2[0],bin3[0],bin4[0],bin5[0],bin6[0]])
    up_unc = np.array([bin1[2],bin2[2],bin3[2],bin4[2],bin5[2],bin6[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1],bin4[1],bin5[1],bin6[1]])
    x = np.array([1.,2.,3.,4.,5.,6.])

    return rt.TGraphAsymmErrors(n,center,x,do_unc,up_unc,np.zeros(6),np.zeros(6));

rt.gStyle.SetOptTitle(0)
rt.gStyle.SetOptStat(0)
rt.gStyle.SetEndErrorSize(0)

c = rt.TCanvas("c", "c", 800, 800)
p1 = rt.TPad("pad1","pad1",0,0.66,1,1)
p2 = rt.TPad("pad2","pad2",0,0,1,0.66)
p1.SetLeftMargin(0.3)
p2.SetLeftMargin(0.3)
p1.SetBottomMargin(0.25)
p2.SetTopMargin(0)
p1.SetTopMargin(0.15)
#    p1.SetRightMargin(0.05)
#    p2.SetRightMargin(0.05)
#    p1.SetBorderMode(0)
#    p2.SetBorderMode(0)
p1.Draw()
p2.Draw()

textsize1 = 0.1
textsize2 = 0.05

lumi = 138
tag1 = rt.TLatex(0.68, 0.88, "%.0f fb^{-1} (13 TeV)" % lumi)
tag1.SetNDC()
tag1.SetTextFont(42)
tag1.SetTextSize(textsize1)
tag2 = rt.TLatex(0.3, 0.88, "#bf{CMS}")
tag2.SetNDC()
tag2.SetTextFont(42)
tag3 = rt.TLatex(0.38, 0.88, "#it{Preliminary}")
tag3.SetNDC()
tag3.SetTextFont(42)
tag2.SetTextSize(textsize1)
tag3.SetTextSize(textsize1)

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

p1.cd()
h1 = rt.TH1D("dummy1","dummy1",2,0.5,2.5)
h1.Fill("1000 < m_{jj} < 2000 GeV",-15)
h1.Fill("m_{jj} > 2000 GeV",-15)
h1.SetLineColor(0)
h1.GetXaxis().SetTitle("VBF category")
h1.GetXaxis().SetTitleSize(textsize1)
h1.GetXaxis().SetTitleOffset(1.5)
h1.GetXaxis().SetLabelSize(textsize1)
h1.GetYaxis().SetTitle('#mu_{VBF}')
h1.GetYaxis().SetTitleSize(textsize1)
h1.GetYaxis().SetTitleOffset(0.75)
h1.GetYaxis().SetLabelSize(textsize1)
h1.GetYaxis().SetRangeUser(-15,15)
h1.Draw("hbar")

sm.Draw("lsame")
#    zero.Draw("lsame")

gvbfunc = get_total_uncertainty("VBF","logs/fit_batch.out")
gvbfunc.SetLineColor(4)
gvbfunc.SetFillColor(4)
gvbfunc.SetFillStyle(3003)
gvbfunc.SetLineWidth(3)
gvbfunc.Draw("3same")

gvbf = get_total("VBF")
gvbf.SetLineColor(4)
gvbf.SetFillColor(4)
gvbf.SetFillStyle(3003)
gvbf.SetLineWidth(3)
gvbf.Draw("lsame")

gvbf3 = get_VBF_table("../allyears-bybin/logs/fit_batch_stat.out")
gvbf3.SetMarkerSize(0)
gvbf3.SetLineColor(94)
gvbf3.SetLineWidth(20)
gvbf3.Draw("psame")

gvbf2 = get_VBF_table("../allyears-bybin/logs/fit_batch.out")
gvbf2.SetMarkerColor(1)
gvbf2.SetMarkerStyle(20)
gvbf2.SetLineColor(1)
gvbf2.SetLineWidth(3)
gvbf2.Draw("psame")

tag1.Draw()
tag2.Draw()
#    tag3.Draw()

p2.cd()
h2 = rt.TH1D("dummy2","dummy2",6,0.5,6.5)
h2.Fill("450 < p_{T} < 500 GeV",-15)
h2.Fill("500 < p_{T} < 550 GeV",-15)
h2.Fill("550 < p_{T} < 600 GeV",-15)
h2.Fill("600 < p_{T} < 675 GeV",-15)
h2.Fill("675 < p_{T} < 800 GeV",-15)
h2.Fill("800 < p_{T} < 1200 GeV",-15)
h2.SetLineColor(0) 
h2.GetXaxis().SetTitle("ggF category")
h2.GetXaxis().SetTitleSize(textsize2)
h2.GetXaxis().SetLabelSize(textsize2)
h2.GetXaxis().SetTitleOffset(3)
h2.GetYaxis().SetTitle('#mu_{ggF}')
h2.GetYaxis().SetTitleSize(textsize2)
h2.GetYaxis().SetLabelSize(textsize2)
h2.GetYaxis().SetTitleOffset(0.75)
h2.GetYaxis().SetRangeUser(-15,15)
h2.Draw("hbar")

sm.Draw("lsame")
#    zero.Draw("lsame")

gggfunc = get_total_uncertainty("ggF","logs/fit_batch.out")
gggfunc.SetLineColor(4)
gggfunc.SetFillColor(4)
gggfunc.SetFillStyle(3003)
gggfunc.SetLineWidth(3)
gggfunc.Draw("3same")

gggf = get_total("ggF")
gggf.SetLineColor(4)
gggf.SetFillColor(4)
gggf.SetFillStyle(3003)
gggf.SetLineWidth(3)
gggf.Draw("lsame")

gggf3 = get_ggF_table("../allyears-bybin/logs/fit_batch_stat.out")
gggf3.SetMarkerSize(0)
gggf3.SetLineColor(94)
gggf3.SetLineWidth(20)
gggf3.Draw("psame")

gggf2 = get_ggF_table("../allyears-bybin/logs/fit_batch.out")
gggf2.SetMarkerColor(1)
gggf2.SetFillColor(94)
gggf2.SetMarkerStyle(20)
gggf2.SetLineColor(1)
gggf2.SetLineWidth(3)
gggf2.Draw("psame")

leg = rt.TLegend(0.33, 0.35, 0.6, 0.82)
leg.SetBorderSize(0)
leg.SetTextFont(42)
leg.SetFillColor(rt.kWhite)
leg.SetLineColor(rt.kWhite)
leg.SetLineStyle(0)
leg.SetFillStyle(0)
leg.SetLineWidth(0)

gggf4 = gggf2.Clone()
gggf4.SetLineWidth(0)

leg.AddEntry(gggf2,"Observed","p")
leg.AddEntry(gggf2,"#pm1#sigma (stat #oplus syst)","l")
leg.AddEntry(gggf4,"#pm1#sigma (stat)","f")
leg.AddEntry(gggf,"Combined fit","lf")
#leg.AddEntry(sm,"SM expectation","l")

p1.cd()
leg.Draw("same")

c.Print("mu_bybin.pdf")
c.Print("mu_bybin.png")
c.Print("mu_bybin.C")

c.Draw()
