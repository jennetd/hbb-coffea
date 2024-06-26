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
  search_string = "r"+poi+" : "

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

def get_VBF_table(vbf_logfile, normalized=False):

  bin8 = read_from_file("VBF1000to1500",vbf_logfile)*smxs['VBF'][str(1)]['nom']
  bin9 = read_from_file("VBF1500plus",vbf_logfile)*smxs['VBF'][str(2)]['nom']

  if normalized:
    bin9 = read_from_file("VBF1500plus",vbf_logfile)
    bin8 = read_from_file("VBF1000to1500",vbf_logfile)

  print(bin8)
  print(bin9)

  n = 2
  center = np.array([bin8[0],bin9[0]])
  up_unc = np.array([bin8[2],bin9[2]])
  do_unc = np.array([bin8[1],bin9[1]])
  x = np.array([1250.,3000.])
  w = np.array([250.,1500.])

  return rt.TGraphAsymmErrors(n,x,center,w,w,do_unc,up_unc)

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--poi", help="POI for 1D likelihood scan")

    args = argParser.parse_args()
    poi = args.poi

    rt.gStyle.SetOptTitle(0)
    rt.gStyle.SetOptStat(0)
    rt.gStyle.SetEndErrorSize(0)

    c = rt.TCanvas("vbfxs", "vbfxs", 600, 600)
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

    h1 = rt.TH1D("dummy1","dummy1",2,np.array([1000.,1500.,4500.]))

    h1.SetLineColor(0)
    h1.SetLineWidth(3)
    h1.GetXaxis().SetTitle("m_{jj}^{gen} [GeV]")
    h1.GetXaxis().SetTitleSize(textsize1)
    h1.GetXaxis().SetLabelSize(textsize1)
    h1.GetXaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
    h1.GetYaxis().SetTitle('#sigma (fb)')
    h1.GetYaxis().SetTitleSize(textsize1)
    h1.GetYaxis().SetLabelSize(textsize1)
    h1.GetYaxis().SetTitleOffset(2*pad1.GetAbsHNDC())
    h1.GetYaxis().SetRangeUser(15,900)
    h1.Draw()

    # Fill SM uncertainties
    sm_center = []
    sm_up = []
    sm_do = []

    for i in range(1,3):
      sm_center += [smxs['VBF'][str(i)]['nom']]
      sm_up += [smxs['VBF'][str(i)]['nom']*(smxs['VBF'][str(i)]['up'])]
      sm_do += [smxs['VBF'][str(i)]['nom']*(smxs['VBF'][str(i)]['down'])]

    x = np.array([1250.,3000.])
    w = np.array([250.,1500.])

    g1 = rt.TGraphAsymmErrors(2,x,np.array(sm_center),w,w,np.array(sm_do),np.array(sm_up))
    g1.SetFillColorAlpha(4,0.1)
    g1.SetFillStyle(3003)
    g1.SetLineColor(4)
    g1.SetLineWidth(3)
    g1.Draw("2same")
    g1.Draw("pesame")

    gvbf = get_VBF_table(vbf_logfile = "logs-obs/fit_batch.out")
    get_VBF_table(vbf_logfile = "logs-obs/fit_batch_stat.out")
    gvbf.SetMarkerColor(1)
    gvbf.SetMarkerStyle(20)
    gvbf.SetLineColor(1)
    gvbf.SetLineWidth(3)
    gvbf.Draw("psame")

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

    leg.AddEntry(g1,"SM (Powheg)","f")
    leg.AddEntry(gvbf,"Data","p")

    leg.Draw("same")

    pad2.cd()

    h2 = h1.Clone("dummy2")
    h2.Reset()
    h2.GetYaxis().SetTitle("Ratio to SM")
    h2.GetYaxis().SetTitleOffset(2*pad2.GetAbsHNDC())
    h2.GetYaxis().SetRangeUser(0,16)
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

    grat = get_VBF_table(vbf_logfile = "logs-obs/fit_batch.out", normalized=True)
    grat.SetMarkerColor(1)
    grat.SetMarkerStyle(20)
    grat.SetLineColor(1)
    grat.SetLineWidth(3)
    grat.Draw("psame")

    c.Print("vbfxs.pdf")
    c.Print("vbfxs.png")
    c.Print("vbfxs.C")

