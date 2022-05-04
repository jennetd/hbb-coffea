/************************************************
 * Jennet Dickinson 
 * Nov 19, 2020
 * Draw Roofit plots
 ************************************************/
#include <iostream>

using namespace RooFit;
using namespace RooStats;

bool blind = true;

void draw(int index, bool pass, bool log=true){

  // Get the year and prefit/postfit/obs from the running directory
  string thisdir = gSystem->pwd();

  string year = "2017";
  string year_string = "41.5/fb, 2017";
  double rZbb = 1;

  // MC only in my case
  string asimov = "MC only";

  // Dummy variable to select the data branch
  string leading_name  = (pass) ? "pass_mv1_": "fail_mv1_";
  string name = (pass) ? "data_pass_2017": "data_fail_2017";

  string filename = "../2017/1mv-signalregion.root";
  TFile *f = new TFile(filename.c_str()); // Can use dataf and read all the distributions from there

  // Root specific stuff (can copy these for later use)
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);

  TCanvas* c = new TCanvas(name.c_str(),name.c_str(),600,600);
  TPad *pad1 = new TPad("pad1","pad1",0,.33,1,1); //pad = python subplot
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

  pad1->cd();
  if( log ) pad1->SetLogy();

  /*DATA*/
  //TFile* dataf = new TFile(filename.c_str());
  TH1D* data_obs;
  data_obs = (TH1D*)f->Get((leading_name+"data_nominal").c_str());

  // blind data!
  if( blind && pass ){                                                                                        
    for(int i=10; i<15; i++){
      data_obs->SetBinContent(i,0);
      data_obs->SetBinError(i,0);
    }                            
  } 

  // Plot data
  data_obs->SetLineColor(kBlack);
  data_obs->SetMarkerColor(kBlack);
  data_obs->SetMarkerStyle(20);    

  // >>>>>>>>>>>Signal<<<<<<<<<<<<
  /* WH */
  TH1D* WH = (TH1D*)f->Get((leading_name+"WH"+"_nominal").c_str()); // Reformat to get from the right file. 
  WH->SetLineColor(kGreen+1);
  WH->SetMarkerColor(kGreen+1);
  WH->SetLineWidth(3);

  /* ZH */
  TH1D* ZH = (TH1D*)WH->Clone("ZH"); // Copy WH and give it ZH name and empty it. 
  ZH->Reset();
  ZH->Add((TH1D*)f->Get((leading_name+"ZH"+"_nominal").c_str()));
  ZH->SetLineColor(kRed+1);
  ZH->SetMarkerColor(kRed+1);
  ZH->SetLineStyle(2);
  ZH->SetLineWidth(3);

  // >>>>>>>>>>> Back ground <<<<<<<<<
  /* bkg Higgs */
  TH1D* bkgHiggs = (TH1D*)WH->Clone("bkgHiggs");
  bkgHiggs->Reset();
  bkgHiggs->Add((TH1D*)f->Get((leading_name+"ggF"+"_nominal").c_str())); // Is this right?
  bkgHiggs->Add((TH1D*)f->Get((leading_name+"VBF"+"_nominal").c_str()));
  bkgHiggs->Add((TH1D*)f->Get((leading_name+"ttH"+"_nominal").c_str()));
  bkgHiggs->SetLineWidth(1);
  bkgHiggs->SetLineColor(kBlack);
  bkgHiggs->SetFillColor(kOrange);

  THStack *bkg = new THStack("bkg",""); //Histogram stack, order add -> stack

  /* VV */
  TH1D* VV = (TH1D*)WH->Clone("VV");
  VV->Reset();
  VV->Add((TH1D*)f->Get((leading_name+"VV"+"_nominal").c_str())); 
  VV->SetLineWidth(1);
  VV->SetLineColor(kBlack);
  VV->SetFillColor(kOrange-3);

  /* single t */
  TH1D* singlet = (TH1D*)WH->Clone("singlet");
  singlet->Reset();
  singlet->Add((TH1D*)f->Get((leading_name+"singlet"+"_nominal").c_str()));
  singlet->SetLineWidth(1);
  singlet->SetLineColor(kBlack);
  singlet->SetFillColor(kPink+6);

  /* ttbar */
  TH1D* ttbar = (TH1D*)f->Get((leading_name+"ttbar"+"_nominal").c_str());
  ttbar->SetLineColor(kBlack);
  ttbar->SetFillColor(kViolet-5);

  /* Z + jets */
  TH1D* Zjets = (TH1D*)f->Get((leading_name+"Zjets"+"_nominal").c_str());
  Zjets->SetLineColor(kBlack);
  Zjets->SetFillColor(kAzure+8);

  /* Z(bb) + jets */
  TH1D* Zjetsbb = (TH1D*)f->Get((leading_name+"Zjetsbb"+"_nominal").c_str());
  Zjetsbb->Scale(rZbb);
  Zjetsbb->SetLineColor(kBlack);
  Zjetsbb->SetFillColor(kAzure-1);

  /* W + jets */
  TH1D* Wjets = (TH1D*)f->Get((leading_name+"Wjets"+"_nominal").c_str());
  Wjets->SetLineColor(kBlack);
  Wjets->SetFillColor(kGray);
  
  
  /* QCD */
  TH1D* qcd = (TH1D*)f->Get((leading_name+"QCD"+"_nominal").c_str());
  qcd->SetLineColor(kBlack);
  qcd->SetFillColor(kWhite);

  if( log ){
    bkg->Add(bkgHiggs);
    bkg->Add(VV);
    bkg->Add(singlet);
    bkg->Add(ttbar);
    bkg->Add(Zjets);
    bkg->Add(Zjetsbb);
    bkg->Add(Wjets);
    bkg->Add(qcd);
  }
  else{
    bkg->Add(qcd);
    bkg->Add(Wjets);
    bkg->Add(Zjetsbb);
    bkg->Add(Zjets);
    bkg->Add(ttbar);
    bkg->Add(singlet);
    bkg->Add(VV);
    bkg->Add(bkgHiggs);
  }

  cout << "QCD: "     << qcd->Integral()     << endl;
  cout << "Wjets: "   << Wjets->Integral()   << endl;
  cout << "Zjets: "   << Zjets->Integral()   << endl;
  cout << "ttbar: "   << ttbar->Integral()   << endl;
  cout << "singlet: " << singlet->Integral() << endl;
  cout << "VV: "      << VV->Integral()      << endl;
  cout << "bkgHiggs: " << bkgHiggs->Integral() << endl;

  bkg->Draw("hist");
  WH->Draw("histsame");
  ZH->Draw("histsame");
  data_obs->Draw("pesame");
  data_obs->Draw("axissame");
  
  double x1=.6, y1=.88;
  TLegend* leg = new TLegend(x1,y1,x1+.3,y1-.3);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetNColumns(2);
  leg->SetTextSize(textsize1);

  leg->AddEntry(data_obs,"Data","p");
  leg->AddEntry(qcd,"QCD","f");
  leg->AddEntry(Wjets,"W","f");
  leg->AddEntry(Zjets,"Z(qq)","f");
  leg->AddEntry(Zjetsbb,"Z(bb)","f");
  leg->AddEntry(ttbar,"t#bar{t}","f");
  leg->AddEntry(singlet,"Single t","f");
  leg->AddEntry(VV,"VV","f");
  leg->AddEntry(bkgHiggs,"Bkg. H","f");
  leg->AddEntry(ZH,"ZH","l");
  leg->AddEntry(WH,"WH","l");


  leg->Draw();

  TLatex l1;
  l1.SetNDC();
  l1.SetTextFont(42);
  l1.SetTextSize(textsize1);
  l1.DrawLatex(0.2,.92,"CMS Preliminary");

  TLatex l2;
  l2.SetNDC();
  l2.SetTextFont(42);
  l2.SetTextSize(textsize1);
  l2.DrawLatex(0.7,.92,year_string.c_str());

  TLatex l3;
  l3.SetNDC();
  l3.SetTextFont(42);
  l3.SetTextSize(textsize1);

  string text = "DDC fail";
  if( pass )
    text = "DDC pass";

  l3.DrawLatex(0.2,.82,text.c_str());

  TLatex l4;
  l4.SetNDC();
  l4.SetTextFont(42);
  l4.SetTextSize(textsize1);
  //string text2 = "p_{T}^{H} bin "+to_string(index+1);  
  //l4.DrawLatex(0.2,.75,text2.c_str());
  
  pad2->cd();

  TH1D* WH_sub = (TH1D*)WH->Clone("WH_sub");
  WH_sub->Reset();
  TH1D* ZH_sub = (TH1D*)ZH->Clone("ZH_sub");
  ZH_sub->Reset();

  WH_sub->Draw("histsame");                                                                                                
  ZH_sub->Draw("histsame");                                                                                                

  if( !log ) name += "_lin";

  c->SaveAs(("plots/"+name+".png").c_str());
  c->SaveAs(("plots/"+name+".pdf").c_str());

  return;

}

void draw_datafit(){

  draw(0,0,0);
  draw(0,1,0);

  return 0;

}
