/************************************************
 * Jennet Dickinson 
 * Nov 19, 2020
 * Draw Roofit plots
 ************************************************/
#include <iostream>

using namespace RooFit;
using namespace RooStats;

bool blind = false;

void muoncr(bool pass, string year="0", bool log=true){

  // Get the year and prefit/postfit/obs from the running directory                                                                                                      
  string thisdir = gSystem->pwd();

  string year_string = "0";
  double rZbb = 1;

  if(year == "2016APV")
    year_string = "19.5/fb, 2016 APV";
  else if(year == "2016")
    year_string = "16.8/fb, 2016";
  else if(year == "2017")
    year_string = "41.5/fb, 2017";
  else if(year == "2018")
    year_string = "59.2/fb, 2018";

  string asimov = "Observed";
  if(thisdir.find("postfit") != std::string::npos)
    asimov = "postfit";
  if(thisdir.find("prefit") !=std::string::npos)
    asimov = "prefit";

  /* DATA */
  TFile* dataf = new TFile(("../"+year+"/muonCR.root").c_str());
  TH1D* data_obs_0;
  if( pass )
    data_obs_0 = (TH1D*)dataf->Get("pass_muondata_nominal");
  else
    data_obs_0 = (TH1D*)dataf->Get("fail_muondata_nominal");

  data_obs_0->Rebin(data_obs_0->GetNbinsX());

  string filename = "fitDiagnosticsTest.root";
  string name = "muonCRfail"+year;
  if( pass )
    name = "muonCRpass"+year;

  string histdirname = "shapes_fit_s/" + name+ "/";

  cout << histdirname << endl;

  TFile *f = new TFile(filename.c_str());

  /* ttbar */

  cout << histdirname << endl;
  TH1D* ttbar = (TH1D*)f->Get((histdirname+"ttbar").c_str());
  ttbar->SetLineColor(kBlack);
  ttbar->SetFillColor(kViolet-5);

  TH1D* data_obs = (TH1D*)ttbar->Clone("data_obs");
  data_obs->Reset();
  data_obs->SetBinContent(1,data_obs_0->GetBinContent(1));
  data_obs->SetBinError(1,data_obs_0->GetBinError(1));
  data_obs->SetFillColor(0);
  data_obs->SetLineColor(kBlack);
  data_obs->SetMarkerColor(kBlack);
  data_obs->SetMarkerStyle(20);

  /* VV */
  TH1D* VV = (TH1D*)ttbar->Clone("VV");
  VV->Reset();
  VV->Add((TH1D*)f->Get((histdirname+"VV").c_str()));
  VV->SetLineWidth(1);
  VV->SetLineColor(kBlack);
  VV->SetFillColor(kOrange-3);

  /* single t */
  TH1D* singlet = (TH1D*)ttbar->Clone("singlet");
  singlet->Reset();
  singlet->Add((TH1D*)f->Get((histdirname+"singlet").c_str()));
  singlet->SetLineWidth(1);
  singlet->SetLineColor(kBlack);
  singlet->SetFillColor(kPink+6);

  /* Z + jets */
  TH1D* Zjets = (TH1D*)ttbar->Clone("Zjets");
  Zjets->Reset();
  Zjets->Add((TH1D*)f->Get((histdirname+"Zjets").c_str()));
  Zjets->Add((TH1D*)f->Get((histdirname+"EWKZ").c_str()));
  Zjets->SetLineColor(kBlack);
  Zjets->SetFillColor(kAzure+8);

  /* Z(bb) + jets */
  TH1D* Zjetsbb = (TH1D*)ttbar->Clone("Zjetsbb");
  Zjetsbb->Reset();
  Zjetsbb->Add((TH1D*)f->Get((histdirname+"Zjetsbb").c_str()));
  Zjetsbb->Scale(rZbb);
  Zjetsbb->SetLineColor(kBlack);
  Zjetsbb->SetFillColor(kAzure-1);

  /* W + jets */
  TH1D* Wjets = (TH1D*)ttbar->Clone("Wjets");
  Wjets->Reset();
  Wjets->Add((TH1D*)f->Get((histdirname+"Wjets").c_str()));
  Wjets->Add((TH1D*)f->Get((histdirname+"EWKW").c_str()));
  Wjets->SetLineColor(kBlack);
  Wjets->SetFillColor(kGray);

  /* QCD */
  TH1D* qcd = (TH1D*)ttbar->Clone();
  qcd->Reset();
  qcd->Add((TH1D*)f->Get((histdirname+"QCD").c_str()));
  qcd->SetLineColor(kBlack);
  qcd->SetFillColor(kWhite);

  /* total background */
  TH1D* TotalBkg = (TH1D*)f->Get((histdirname+"/total_background").c_str());
  TotalBkg->SetMarkerColor(kRed);
  TotalBkg->SetLineColor(kRed);
  TotalBkg->SetFillColor(kRed);
  TotalBkg->SetFillStyle(3003);

  double max = TotalBkg->GetMaximum();
  TotalBkg->GetYaxis()->SetRangeUser(0.1,1000*max);
  if( !log ) TotalBkg->GetYaxis()->SetRangeUser(0,1.3*max);
  TotalBkg->GetYaxis()->SetTitle("Events / 7 GeV");
  TotalBkg->GetXaxis()->SetTitle("m_{sd} [GeV]");

  THStack *bkg = new THStack("bkg","");
  if( log ){
    bkg->Add(VV);
    bkg->Add(Zjets);
    bkg->Add(Zjetsbb);
    bkg->Add(Wjets);
    bkg->Add(qcd);
    bkg->Add(singlet);
    bkg->Add(ttbar);
  }
  else{
    bkg->Add(ttbar);
    bkg->Add(singlet);
    bkg->Add(qcd);
    bkg->Add(Wjets);
    bkg->Add(Zjetsbb);
    bkg->Add(Zjets);
    bkg->Add(VV);
  }

  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);

  TCanvas* c = new TCanvas(name.c_str(),name.c_str(),600,600);
  TPad *pad1 = new TPad("pad1","pad1",0,.33,1,1);
  TPad *pad2 = new TPad("pad2","pad2",0,0,1,.33);

  pad1->SetBottomMargin(0.00001);
  pad1->SetTopMargin(0.1);
  pad1->SetBorderMode(0);
  pad2->SetTopMargin(0.00001);
  pad2->SetBottomMargin(0.3);
  pad2->SetBorderMode(0);

  pad1->SetLeftMargin(0.15);
  pad2->SetLeftMargin(0.15);
  pad1->Draw();
  pad2->Draw();

  float textsize1 = 16/(pad1->GetWh()*pad1->GetAbsHNDC());
  float textsize2 = 16/(pad2->GetWh()*pad2->GetAbsHNDC());

  TotalBkg->GetYaxis()->SetTitleSize(textsize1);
  TotalBkg->GetYaxis()->SetLabelSize(textsize1);
  TotalBkg->GetYaxis()->SetTitleOffset(2*pad1->GetAbsHNDC());

  pad1->cd();
  if( log ) pad1->SetLogy();

  //  cout << "QCD: "     << qcd->Integral()     << endl;
  //  cout << "Wjets: "   << Wjets->Integral()   << endl;
  //  cout << "Zjets: "   << Zjets->Integral()   << endl;
  //  cout << "ttbar: "   << ttbar->Integral()   << endl;
  //  cout << "singlet: " << singlet->Integral() << endl;
  //  cout << "VV: "      << VV->Integral()      << endl;
  cout << "total bkg: " << TotalBkg->Integral() << endl;
  cout << "total bkg down: " << TotalBkg->GetBinError(1) << endl;
  cout << "data: " << data_obs->Integral() << endl;


  TotalBkg->Draw("e2");
  bkg->Draw("histsame");
  TotalBkg->Draw("e2same");
  data_obs->Draw("pesame");
  data_obs->Draw("axissame");

  double x1=.6, y1=.86;
  TLegend* leg = new TLegend(x1,y1,x1+.3,y1-.26);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetNColumns(2);
  leg->SetTextSize(textsize1);

  leg->AddEntry(data_obs,"Data","p");
  leg->AddEntry(TotalBkg,"Bkg. Unc.","f");
  leg->AddEntry(qcd,"QCD","f");
  leg->AddEntry(Wjets,"W","f");
  leg->AddEntry(Zjets,"Z(qq)","f");
  leg->AddEntry(Zjetsbb,"Z(bb)","f");
  leg->AddEntry(ttbar,"t#bar{t}","f");
  leg->AddEntry(singlet,"Single t","f");
  leg->AddEntry(VV,"VV","f");

  leg->Draw();

  TLatex l1;
  l1.SetNDC();
  l1.SetTextFont(42);
  l1.SetTextSize(textsize1);
  l1.DrawLatex(0.2,.82,"#bf{CMS} Preliminary");

  TLatex l2;
  l2.SetNDC();
  l2.SetTextFont(42);
  l2.SetTextSize(textsize1);
  l2.DrawLatex(0.7,.92,year_string.c_str());

  TLatex l3;
  l3.SetNDC();
  l3.SetTextFont(42);
  l3.SetTextSize(textsize1);
  string text = "DeepDoubleB Fail Region";
  if( pass )
    text = "DeepDoubleB Pass Region";
  l3.DrawLatex(0.2,.77,text.c_str());

  TLatex l4;
  l4.SetNDC();
  l4.SetTextFont(42);
  l4.SetTextSize(textsize1);
  string text2 = "Muon CR";
  l4.DrawLatex(0.2,.72,text2.c_str());

  pad2->cd();

  TH1D* TotalBkg_sub = (TH1D*)TotalBkg->Clone("TotalBkg_sub");
  TotalBkg_sub->Reset();
  TH1D* data_obs_sub = (TH1D*)data_obs->Clone("data_obs_ratio");
  data_obs_sub->Reset();

  for(int i=1; i<TotalBkg_sub->GetNbinsX()+1; i++){
    TotalBkg_sub->SetBinError(i,TotalBkg->GetBinError(i)/data_obs->GetBinError(i));

    data_obs_sub->SetBinContent(i,(data_obs->GetBinContent(i)-TotalBkg->GetBinContent(i))/data_obs->GetBinError(i));
    data_obs_sub->SetBinError(i,data_obs->GetBinError(i)/data_obs->GetBinError(i));
  }

  TotalBkg_sub->GetYaxis()->SetTitleSize(textsize2);
  TotalBkg_sub->GetYaxis()->SetLabelSize(textsize2);
  TotalBkg_sub->GetXaxis()->SetTitleSize(textsize2);
  TotalBkg_sub->GetXaxis()->SetLabelSize(textsize2);
  TotalBkg_sub->GetYaxis()->SetTitleOffset(2*pad2->GetAbsHNDC());
  TotalBkg_sub->GetYaxis()->SetTitle("(Data - Bkg)/#sigma_{Data}");
  TotalBkg_sub->SetMarkerSize(0);

  double min2 = data_obs_sub->GetMinimum();
  double max2 = data_obs_sub->GetMaximum();
  if( !pass ){
    max2 += 1;
    min2 -= 1;
  }
  TotalBkg_sub->GetYaxis()->SetRangeUser(1.3*min2,1.3*max2);

  TotalBkg_sub->Draw("e2");
  data_obs_sub->Draw("pesame");

  if( !log ) name += "_lin";

  c->SaveAs(("plots/"+name+".png").c_str());
  c->SaveAs(("plots/"+name+".pdf").c_str());

  return;

}

void draw_muon(){

  muoncr(0,"2016APV",0);                                                                                                
  muoncr(1,"2016APV",0); 
  muoncr(0,"2016",0);
  muoncr(1,"2016",0);
  muoncr(0,"2017",0);
  muoncr(1,"2017",0);
  muoncr(0,"2018",0);
  muoncr(1,"2018",0);

  return 0;

}
