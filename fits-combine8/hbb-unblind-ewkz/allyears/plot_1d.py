import ROOT as rt
from array import array
from scipy.interpolate import Rbf
import numpy as np
import glob
import pandas as pd
import argparse

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--poi", help="POI for 1D likelihood scan")

    args = argParser.parse_args()
    poi = args.poi

    rt.gStyle.SetOptTitle(0)
    rt.gStyle.SetOptStat(0)
    rt.gStyle.SetPadRightMargin(0.15)
    rt.gStyle.SetPalette(rt.kBird)
    rt.gStyle.SetNumberContours(999)

    c = rt.TCanvas("c", "c", 500, 400)

    # syst + stat
    limit = rt.TChain("limit")
    for ifile in glob.glob("higgsCombiner"+poi+".Total.root"):
        limit.Add(ifile)

    deltaNLL = []
    mu = []

    for x in limit:
        deltaNLL += [getattr(x, 'deltaNLL')]
        mu +=[ getattr(x, 'r'+poi)]

    df = pd.DataFrame()
    df['mu'] = mu
    df['deltaNLL'] = deltaNLL
    df.drop_duplicates(inplace=True)
    df.sort_values('mu',inplace=True)

    g = rt.TGraph(len(df))
    for i in range(len(df)):
        g.SetPoint(i,df['mu'].iloc[i],2*df['deltaNLL'].iloc[i])
        g.SetMarkerStyle(34)
        g.SetMarkerSize(1.5)
    g.SetLineColor(1)
    g.SetLineWidth(3)

    # stat only
    limit = rt.TChain("limit")
    for ifile in glob.glob("higgsCombiner"+poi+"Stat.Total.root"):
        limit.Add(ifile)

    deltaNLL = []
    mu = []

    for x in limit:
        deltaNLL += [getattr(x, 'deltaNLL')]
        mu +=[ getattr(x, 'r'+poi)]

    df = pd.DataFrame()
    df['mu'] = mu
    df['deltaNLL'] = deltaNLL
    df.drop_duplicates(inplace=True)
    df.sort_values('mu',inplace=True)

    gstat = rt.TGraph(len(df))
    for i in range(len(df)):
        gstat.SetPoint(i,df['mu'].iloc[i],2*df['deltaNLL'].iloc[i])
        gstat.SetMarkerStyle(34)
        gstat.SetMarkerSize(1.5)
    gstat.SetLineColor(4)
    gstat.SetLineStyle(2)
    gstat.SetLineWidth(3)

    g.GetYaxis().SetRangeUser(0,6);
    g.GetXaxis().SetRangeUser(-2,6);
    if "VBF" in poi:
        g.GetXaxis().SetRangeUser(-1,10);
    if "Hbb" in poi:
        g.GetXaxis().SetRangeUser(-1,10);
        g.GetYaxis().SetRangeUser(0,24);
    g.GetXaxis().SetTitle("#mu_"+poi);
    g.GetYaxis().SetTitle("2#DeltaNLL");

    g.Draw('ac')
    gstat.Draw('csame')

    lumi = 138
    tag1 = rt.TLatex(0.65, 0.92, "%.0f fb^{-1} (13 TeV)" % lumi)
    tag1.SetNDC()
    tag1.SetTextFont(42)
    tag1.SetTextSize(0.04)
    tag2 = rt.TLatex(0.13, 0.85, "CMS")
    tag2.SetNDC()
    tag2.SetTextFont(62)
    tag3 = rt.TLatex(0.22, 0.85, "Preliminary")
    tag3.SetNDC()
    tag3.SetTextFont(52)
    tag2.SetTextSize(0.05)
    tag3.SetTextSize(0.04)
    tag1.Draw()
    tag2.Draw()
    tag3.Draw()

    leg = rt.TLegend(0.65, 0.7, 0.85, 0.87)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetFillColor(rt.kWhite)
    leg.SetLineColor(rt.kWhite)
    leg.SetLineStyle(0)
    leg.SetFillStyle(0)
    leg.SetLineWidth(0)
    leg.AddEntry(g, "stat + syst", "l")
    leg.AddEntry(gstat, "stat only", "l")
    leg.Draw("same")

    c.Print("mu_"+poi+".pdf")
    c.Print("mu_"+poi+".C")

#    htemp.Draw("axis")
#    cl68.Draw("L SAME")
#    cl95.Draw("L SAME")
#    fit.Draw("P SAME")
