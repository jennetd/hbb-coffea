import ROOT as rt
from array import array
from scipy.interpolate import Rbf
import numpy as np
import glob
import pandas as pd
import argparse
import json

rt.gStyle.SetOptTitle(0)
rt.gStyle.SetOptStat(0)
rt.gStyle.SetEndErrorSize(0)

with open('xs.json') as f:
    smxs = json.load(f)

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

    return np.array(vals)

def get_ggF_table(logfile,normalized=False):
  
    bin1 = read_from_file("ggF300to450",logfile)*smxs['ggF'][str(1)]['nom']
    bin2 = read_from_file("ggF450to650",logfile)*smxs['ggF'][str(2)]['nom']
    bin3 = read_from_file("ggF650plus",logfile)*smxs['ggF'][str(3)]['nom']

    if normalized:
        bin1 = read_from_file("ggF300to450",logfile)
        bin2 = read_from_file("ggF450to650",logfile)
        bin3 = read_from_file("ggF650plus",logfile)

    n = 3;
    center = np.array([bin1[0],bin2[0],bin3[0]])
    up_unc = np.array([bin1[2],bin2[2],bin3[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1]])
    
    print('ggF xs',center)
    print('ggF up xs',up_unc)
    print('ggF do xs',do_unc)
  
    x = np.linspace(0,n,n+1)
    w = np.array([0 for i in x])
    
    return rt.TGraphAsymmErrors(n,np.array(x),np.array(center),np.array(w),np.array(w),np.array(up_unc),np.array(do_unc))

def get_VBF_table(logfile,normalized=False):

    bin8 = read_from_file("VBF1000to1500",logfile)*smxs['VBF'][str(1)]['nom']
    bin9 = read_from_file("VBF1500plus",logfile)*smxs['VBF'][str(2)]['nom']

    if normalized:
        bin9 = read_from_file("VBF1500plus",logfile)
        bin8 = read_from_file("VBF1000to1500",logfile)

    n = 2;
    center = np.array([bin8[0],bin9[0]])
    up_unc = np.array([bin8[2],bin9[2]])
    do_unc = np.array([bin8[1],bin9[1]])

    print('VBF xs',center)
    print('VBF up xs',up_unc)
    print('VBF do xs',do_unc)

    x = np.linspace(3,3+n,n+1)
    w = np.array([0 for i in x])

    return rt.TGraphAsymmErrors(n,np.array(x),np.array(center),np.array(w),np.array(w),np.array(up_unc),np.array(do_unc))

c = rt.TCanvas("stxs", "stxs", 800, 600)
pad1 = rt.TPad("pad1","pad1",0,.4,0.6,1);
pad2 = rt.TPad("pad2","pad2",0,0,0.6,.4);
pad3 = rt.TPad("pad3","pad3",0.6,.4,1,1);
pad4 = rt.TPad("pad4","pad4",0.6,0,1,.4);

pad1.SetBottomMargin(0.05)
pad1.SetTopMargin(0.1)
pad1.SetBorderMode(0)
pad2.SetTopMargin(0.00001)
pad2.SetBottomMargin(0.3)
pad2.SetBorderMode(0)
pad3.SetBottomMargin(0.05)
pad3.SetTopMargin(0.1)
pad3.SetBorderMode(0)
pad4.SetTopMargin(0.00001)
pad4.SetBottomMargin(0.3)
pad4.SetBorderMode(0)    

pad1.SetLeftMargin(0.15)
pad2.SetLeftMargin(0.15)
pad3.SetRightMargin(0.15)
pad4.SetRightMargin(0.15)
pad1.SetRightMargin(0.02)
pad2.SetRightMargin(0.02)
pad3.SetLeftMargin(0.02)
pad4.SetLeftMargin(0.02)
pad1.Draw()
pad2.Draw()
pad3.Draw()
pad4.Draw()

textsize1 = 19/(pad1.GetWh()*pad1.GetAbsHNDC());
textsize2 = 1.5*textsize1
textsize3 = textsize1
textsize4 = textsize2

pad1.cd()
pad1.SetLogy()

lumi = 138
tag1 = rt.TLatex(0.46, 0.92, "%.0f fb^{-1} (13 TeV)" % lumi)
tag1.SetNDC()
tag1.SetTextFont(42)
tag2 = rt.TLatex(0.17, 0.92, "CMS")
tag2.SetNDC()
tag2.SetTextFont(62)
tag3 = rt.TLatex(0.26, 0.92, "H(bb) STXS stage 1.2")
tag3.SetNDC()
tag3.SetTextFont(42)

tag4 = rt.TLatex(0.2, 0.82, "ggF")
tag4.SetNDC()
tag4.SetTextFont(42)
tag5 = rt.TLatex(0.08, 0.82, "VBF")
tag5.SetNDC()
tag5.SetTextFont(42)

h1 = rt.TH1D("dummy1","dummy1",3,-0.5,2.5)

binnames = ['[300,450]', '[450,650]', '[650,#infty)']
for b in binnames:
    h1.Fill(b,0)

h1.SetLineColor(0)
h1.SetLineWidth(3)
h1.GetXaxis().SetTitleSize(0)
h1.GetXaxis().SetLabelSize(0)
h1.GetYaxis().SetTitle('#sigma_{obs} [fb]')
h1.GetYaxis().SetTitleSize(textsize1)
h1.GetYaxis().SetLabelSize(textsize1)
h1.GetYaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
h1.GetYaxis().SetRangeUser(1,3000)
h1.GetXaxis().SetTitle("p_{T}^{H} [GeV]")
h1.GetXaxis().CenterTitle(True)
h1.GetXaxis().SetTitleOffset(2.5*pad1.GetAbsHNDC())
h1.Draw()

# Fill SM uncertainties
ggf_center = []
ggf_up = []
ggf_do = []
for i in range(1,4):
    ggf_center += [smxs['ggF'][str(i)]['nom']]
    ggf_up += [smxs['ggF'][str(i)]['nom']*(smxs['ggF'][str(i)]['up'])]
    ggf_do += [smxs['ggF'][str(i)]['nom']*(smxs['ggF'][str(i)]['down'])]

x1 = np.linspace(0,3,4)
w1 = np.array([0.5 for i in x1])

g1 = rt.TGraphAsymmErrors(3,x1,np.array(ggf_center),w1,w1,np.array(ggf_do),np.array(ggf_up))
g1.SetFillColor(4)
g1.SetFillStyle(3003)
g1.SetLineColor(4)
g1.SetLineWidth(1)
g1.Draw("2same")
g1.Draw("pesame")

gggf = get_ggF_table(logfile = "logs/fit_batch.out")
print("Stat only")
get_ggF_table(logfile = "logs/fit_batch_stat.out")                                                                                                                        
gggf.SetMarkerColor(1)
gggf.SetMarkerStyle(20)
gggf.SetLineColor(1)
gggf.SetLineWidth(3)
gggf.Draw("pesame")

tag2.SetTextSize(textsize1)
tag2.Draw()
tag3.SetTextSize(textsize1)
tag3.Draw()
tag4.SetTextSize(textsize1)
tag4.Draw()

pad3.cd()
pad3.SetLogy()

h2 = rt.TH1D("dummy2","dummy2",2,2.5,4.5)

binnames = ['[1000,1500]', '[1500,#infty)']
for b in binnames:
    h2.Fill(b,0)

h2.SetLineColor(0)
h2.SetLineWidth(3)
h2.GetXaxis().SetTitleSize(0)
h2.GetXaxis().SetLabelSize(0)
h2.GetXaxis().SetTitle("m_{jj}^{gen} [GeV]")
h2.GetXaxis().CenterTitle(True)
h2.GetXaxis().SetTitleOffset(2.5*pad3.GetAbsHNDC())
h2.GetYaxis().SetRangeUser(1,3000)
h2.GetYaxis().SetTitleSize(0)
h2.GetYaxis().SetLabelSize(0)
h2.Draw()

vbf_center = []
vbf_up = []
vbf_do = []
for i in range(1,3):
    vbf_center += [smxs['VBF'][str(i)]['nom']]
    vbf_up += [smxs['VBF'][str(i)]['nom']*(smxs['VBF'][str(i)]['up'])]
    vbf_do += [smxs['VBF'][str(i)]['nom']*(smxs['VBF'][str(i)]['down'])]

x2 = np.linspace(3,5,3)
w2 = np.array([0.5 for i in x2])

g2 = rt.TGraphAsymmErrors(2,x2,np.array(vbf_center),w2,w2,np.array(vbf_do),np.array(vbf_up))
g2.SetFillColor(94)
g2.SetFillStyle(3003)
g2.SetLineColor(94)
g2.SetMarkerColor(94)
g2.SetLineWidth(1)
g2.Draw("2same")
g2.Draw("pesame")

gvbf = get_VBF_table(logfile = "logs/fit_batch.out")
print("Stat only")
gvbf = get_VBF_table(logfile = "logs/fit_batch_stat.out")
gvbf.SetMarkerColor(1)
gvbf.SetMarkerStyle(20)
gvbf.SetLineColor(1)
gvbf.SetLineWidth(3)
gvbf.Draw("psame")

tag1.SetTextSize(textsize3)
tag1.Draw()
tag5.SetTextSize(textsize3)
tag5.Draw()

pad1.cd()

leg = rt.TLegend(0.44, 0.65, 0.82, 0.87)
leg.SetBorderSize(0)
#    leg.SetTextFont(42)
leg.SetTextSize(textsize3)
leg.SetFillColor(rt.kWhite)
leg.SetLineColor(rt.kWhite)
leg.SetLineStyle(0)
leg.SetFillStyle(0)
leg.SetLineWidth(0)

histo = h2.Clone("histo")
histo.SetMarkerColor(1)
histo.SetLineColor(1)
histo.SetLineWidth(3)
histo.SetMarkerStyle(20)

leg.AddEntry(histo,"Observed (stat #oplus syst)","pe")
leg.AddEntry(g1,"ggF (HJMINLO)","f")
leg.AddEntry(g2,"VBF (POWHEG+HC)","f")

leg.Draw("same")

pad2.cd()

h3 = h1.Clone("dummy3")
h3.Reset()
h3.GetYaxis().SetTitle("#sigma_{obs} / #sigma_{SM}")
h3.GetYaxis().SetTitleOffset(2*pad2.GetAbsHNDC())
h3.GetYaxis().SetRangeUser(-5,16)
h3.GetXaxis().SetTitleSize(textsize2)
h3.GetXaxis().SetLabelSize(1.3*textsize2)
h3.GetYaxis().SetTitleSize(textsize2)
h3.GetYaxis().SetLabelSize(textsize2)
h3.Draw()

ggf_uperrs = [smxs['ggF'][str(i)]["up"] for i in range(1,4)]
ggf_doerrs = [smxs['ggF'][str(i)]["down"] for i in range(1,4)]

g3 = rt.TGraphAsymmErrors(3,x1,np.ones(3),w1,w1,np.array(ggf_doerrs),np.array(ggf_uperrs))
g3.SetFillColor(4)
g3.SetFillStyle(3003)
g3.SetLineColor(4)
g3.SetLineWidth(1)
g3.Draw("2same")
g3.Draw("pesame")

grat = get_ggF_table(logfile = "logs/fit_batch.out",normalized=True)
grat.SetMarkerColor(1)
grat.SetMarkerStyle(20)
grat.SetLineColor(1)
grat.SetLineWidth(3)
grat.Draw("pesame")

pad4.cd()

h4 = h2.Clone("dummy4")
h4.Reset()
h4.GetYaxis().SetTitleSize(0)
h4.GetYaxis().SetLabelSize(0)
h4.GetYaxis().SetRangeUser(-5,16)
h4.GetXaxis().SetTitleSize(textsize2)
h4.GetXaxis().SetLabelSize(1.3*textsize2)
h4.Draw()

vbf_uperrs = [smxs['VBF'][str(i)]["up"] for i in range(1,3)]
vbf_doerrs = [smxs['VBF'][str(i)]["down"] for i in range(1,3)]

g4 = rt.TGraphAsymmErrors(2,x2,np.ones(2),w2,w2,np.array(vbf_doerrs),np.array(vbf_uperrs))
g4.SetFillColor(94)
g4.SetFillStyle(3003)
g4.SetMarkerColor(94)
g4.SetLineColor(94)
g4.SetLineWidth(1)
g4.Draw("2same")
g4.Draw("pesame")

grat2 = get_VBF_table(logfile = "logs/fit_batch.out",normalized=True)
grat2.SetMarkerColor(1)
grat2.SetMarkerStyle(20)
grat2.SetLineColor(1)
grat2.SetLineWidth(3)
grat2.Draw("psame")

c.Print("stxs.pdf")
c.Print("stxs.png")
c.Print("stxs.C")

c.Draw()
