import ROOT as rt
from array import array
from scipy.interpolate import Rbf
import numpy as np
import glob
import pandas as pd
import argparse
import json

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

def get_ggF_table(ggf_logfile,normalized=False):


  bin1 = read_from_file("ggF300to450",ggf_logfile)*smxs['ggF'][str(1)]['nom']
  bin2 = read_from_file("ggF450to650",ggf_logfile)*smxs['ggF'][str(2)]['nom']
  bin3 = read_from_file("ggF650plus",ggf_logfile)*smxs['ggF'][str(3)]['nom']

  print(bin1)
  print(bin2)
  print(bin3)

  if normalized:
      bin1 = read_from_file("ggF300to450",ggf_logfile)
      bin2 = read_from_file("ggF450to650",ggf_logfile)
      bin3 = read_from_file("ggF650plus",ggf_logfile)

  n = 3;
  center = np.array([bin1[0],bin2[0],bin3[0]])
  up_unc = np.array([bin1[2],bin2[2],bin3[2]])
  do_unc = np.array([bin1[1],bin2[1],bin3[1]])

  x = np.array([375.,550.,925.])
  w = np.array([75.,100.,275.])

  return rt.TGraphAsymmErrors(n,x,center,w,w,do_unc,up_unc);

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--poi", help="POI for 1D likelihood scan")

    args = argParser.parse_args()
    poi = args.poi

    rt.gStyle.SetOptTitle(0)
    rt.gStyle.SetOptStat(0)
    rt.gStyle.SetEndErrorSize(0)

    c = rt.TCanvas("ggfxs", "ggfxs", 600, 600)
    pad1 = rt.TPad("pad1","pad1",0,.33,1,1);
    pad2 = rt.TPad("pad2","pad2",0,0,1,.33);

    pad1.SetBottomMargin(0.00001)
    pad1.SetTopMargin(0.1)
    pad1.SetBorderMode(0)
    pad2.SetTopMargin(0.00001)
    pad2.SetBottomMargin(0.3)
    pad2.SetBorderMode(0)
    
    pad1.SetLeftMargin(0.15)
    pad2.SetLeftMargin(0.15)
    pad1.Draw()
    pad2.Draw()

    textsize1 = 16/(pad1.GetWh()*pad1.GetAbsHNDC());
    textsize2 = 16/(pad2.GetWh()*pad2.GetAbsHNDC());

    pad1.cd()
    pad1.SetLogy()

    lumi = 138
    tag1 = rt.TLatex(0.67, 0.92, "%.0f fb^{-1} (13 TeV)" % lumi)
    tag1.SetNDC()
    tag1.SetTextFont(42)
    tag2 = rt.TLatex(0.19, 0.82, "CMS")
    tag2.SetNDC()
    tag2.SetTextFont(62)
    tag3 = rt.TLatex(0.27, 0.82, "Preliminary")
    tag3.SetNDC()
    tag3.SetTextFont(52)

    zerox = np.array([-100.,100.])
    zeroy = np.array([0.,0.])
    zero = rt.TGraph(2,zerox,zeroy)
    zero.SetLineColor(0);
    zero.SetLineWidth(3);

    h1 = rt.TH1D("dummy1","dummy1",3,np.array([300.,450.,650.,1200.]))

    h1.SetLineColor(0)
    h1.SetLineWidth(3)
    h1.GetXaxis().SetTitle("Higgs boson p_{T} [GeV]")
    h1.GetXaxis().SetTitleSize(textsize1)
    h1.GetXaxis().SetLabelSize(textsize1)
    h1.GetXaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
    h1.GetYaxis().SetTitle('#sigma (fb)')
    h1.GetYaxis().SetTitleSize(textsize1)
    h1.GetYaxis().SetLabelSize(textsize1)
    h1.GetYaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
    h1.GetYaxis().SetRangeUser(0.5,5000)
    h1.Draw()

    # Fill SM uncertainties
    sm_center = []
    sm_up = []
    sm_do = []

    for i in range(1,4):
      sm_center += [smxs['ggF'][str(i)]['nom']]
      sm_up += [smxs['ggF'][str(i)]['nom']*(smxs['ggF'][str(i)]['up'])]
      sm_do += [smxs['ggF'][str(i)]['nom']*(smxs['ggF'][str(i)]['down'])]

    x = np.array([375.,550.,925.])
    w = np.array([75.,100.,275.])

    g1 = rt.TGraphAsymmErrors(3,x,np.array(sm_center),w,w,np.array(sm_do),np.array(sm_up))
    g1.SetFillColor(4)
    g1.SetFillStyle(3003)
    g1.SetLineColor(4)
    g1.SetLineWidth(3)
    g1.Draw("2same")
    g1.Draw("pesame")

    gggf = get_ggF_table(ggf_logfile = "logs-obs/fit_batch.out")
    get_ggF_table(ggf_logfile = "logs-obs/fit_batch_stat.out")
    gggf.SetMarkerColor(1)
    gggf.SetMarkerStyle(20)
    gggf.SetLineColor(1)
    gggf.SetLineWidth(3)
    gggf.Draw("psame")

    zero.Draw("lsame")

    tag1.Draw()
    tag2.Draw()
#    tag3.Draw()

    leg = rt.TLegend(0.6, 0.7, 0.85, 0.87)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(textsize1)
    leg.SetFillColor(rt.kWhite)
    leg.SetLineColor(rt.kWhite)
    leg.SetLineStyle(0)
    leg.SetFillStyle(0)
    leg.SetLineWidth(0)

    leg.AddEntry(g1,"SM (HJ-MINLO)","f")
    leg.AddEntry(gggf,"Data","p")

    leg.Draw("same")

    pad2.cd()

    h2 = h1.Clone("dummy2")
    h2.Reset()
    h2.GetYaxis().SetTitle("Ratio to SM")
    h2.GetYaxis().SetTitleOffset(2*pad2.GetAbsHNDC())
    h2.GetYaxis().SetRangeUser(-4.5,11)
    h2.GetXaxis().SetTitleSize(textsize2)
    h2.GetXaxis().SetLabelSize(textsize2)
    h2.GetYaxis().SetTitleSize(textsize2)
    h2.GetYaxis().SetLabelSize(textsize2)
    h2.Draw()

    uperrs = [smxs['ggF'][str(i)]["up"] for i in range(1,4)]
    doerrs = [smxs['ggF'][str(i)]["down"] for i in range(1,4)]

    g2 = rt.TGraphAsymmErrors(3,x,np.ones(3),w,w,np.array(doerrs),np.array(uperrs))
    g2.SetFillColor(4)
    g2.SetFillStyle(3003)
    g2.SetLineColor(4)
    g2.SetLineWidth(0)
    g2.Draw("2same")
    g2.Draw("pesame")

    grat = get_ggF_table(ggf_logfile = "logs-obs/fit_batch.out",normalized=True)
    grat.SetMarkerColor(1)
    grat.SetMarkerStyle(20)
    grat.SetLineColor(1)
    grat.SetLineWidth(3)
    grat.Draw("psame")

    c.Print("ggfxs.pdf")
    c.Print("ggfxs.png")
    c.Print("ggfxs.C")

