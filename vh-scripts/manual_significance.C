/************************************************
 * Jennet Dickinson 
 * Nov 19, 2020
 * Draw Roofit plots
 ************************************************/
#include <iostream>
#include <math.h> 

using namespace RooFit;
using namespace RooStats;

bool blind = true;

//To calculate significance
double significance(double s, double b){
  if (b==0) return 0;

  double z_squared = 2.0*(s+b)*log(1.0+1.0*s/b) - 2.0*s;
  double z =  sqrt(z_squared);

  //double z = s/sqrt(b);
    
  return z;
}

void draw(int pt_index, bool charm, bool pass,  bool log=true){

  // Get the year and prefit/postfit/obs from the running directory
  string thisdir = gSystem->pwd();

  string year = "2016";
  string year_string = "35.9/fb, 2016";
  double rZbb = 1;

  if(thisdir.find("2017") != std::string::npos){
    year = "2017";
    year_string = "41.5/fb, 2017";
    rZbb = 1;
  }
  if(thisdir.find("2018") != std::string::npos){
    year = "2018";
    year_string = "59.9/fb, 2018";
    rZbb = 1;
  }

  // MC only in my case
  string asimov = "MC only";

  //Fit root file
  string filename = "fitDiagnostics.root";

  // branch name
  string name = "ptbin" + to_string(pt_index);
  //Category
  if (charm) name = name + "charm";
  else name = name + "light";
  //Pass fail
  if (pass) name = name + "pass" + year;
  else name = name + "fail" + year;

  //Fit directory
  string hist_dir = "shapes_fit_s/" + name+ "/";
  cout << hist_dir << endl;

  // Dummy variable to select the data branch
  TFile *f = new TFile(filename.c_str()); // Can use dataf and read all the distributions from there

  // >>>>>>>>>>>Signal<<<<<<<<<<<<
  /* WH */
  TH1D* WH = (TH1D*)f->Get((hist_dir+"WH").c_str()); // Reformat to get from the right file.

  cout << hist_dir+"WH" << endl;
  /* ZH */
  TH1D* ZH = (TH1D*)WH->Clone("ZH"); // Copy WH and give it ZH name and empty it. 
  ZH->Reset();
  ZH->Add((TH1D*)f->Get((hist_dir+"ZH").c_str()));

  // >>>>>>>>>>> Back ground <<<<<<<<<
  /* bkg Higgs */
  TH1D* bkgHiggs = (TH1D*)WH->Clone("bkgHiggs");
  bkgHiggs->Reset();
  bkgHiggs->Add((TH1D*)f->Get((hist_dir+"ggF").c_str())); // Is this right?
  bkgHiggs->Add((TH1D*)f->Get((hist_dir+"VBF").c_str()));
  bkgHiggs->Add((TH1D*)f->Get((hist_dir+"ttH").c_str()));

  /* VV */
  TH1D* VV = (TH1D*)WH->Clone("VV");
  VV->Reset();
  VV->Add((TH1D*)f->Get((hist_dir+"VV").c_str())); 

  /* single t */
  TH1D* singlet = (TH1D*)WH->Clone("singlet");
  singlet->Reset();
  singlet->Add((TH1D*)f->Get((hist_dir+"singlet").c_str()));

  /* ttbar */
  TH1D* ttbar = (TH1D*)f->Get((hist_dir+"ttbar").c_str());
  /* Z + jets */
  TH1D* Zjets = (TH1D*)f->Get((hist_dir+"Zjets").c_str());
  /* Z(bb) + jets */
  TH1D* Zjetsbb = (TH1D*)f->Get((hist_dir+"Zjetsbb").c_str());
  /* W + jets */
  TH1D* Wjets = (TH1D*)f->Get((hist_dir+"Wjets").c_str());
  /* QCD */
  TH1D* qcd = (TH1D*)f->Get((hist_dir+"qcd").c_str());

  cout << "QCD: "     << qcd->Integral()     << endl;
  cout << "Wjets: "   << Wjets->Integral()   << endl;
  cout << "Zjets: "   << Zjets->Integral()   << endl;
  cout << "ttbar: "   << ttbar->Integral()   << endl;
  cout << "singlet: " << singlet->Integral() << endl;
  cout << "VV: "      << VV->Integral()      << endl;
  cout << "bkgHiggs: " << bkgHiggs->Integral() << endl;

  double s =  WH->Integral() + ZH->Integral();
  double b = qcd->Integral() + Wjets->Integral() + Zjets->Integral() + ttbar->Integral()*0.1 + singlet->Integral() +  VV->Integral() + bkgHiggs->Integral();

  double z = significance(s,b);

  cout << "Significance for " << name + ": " << z << endl;

  return;

}
    

void manual_significance(){

 //Loop over pt bins
  for(int i=0; i<1; i++){
    draw(i,1,0,0); //charm fail
    draw(i,0,1,0); //charm pass
    draw(i,0,0,0); //light fail 
    draw(i,1,1,0); //light pass
  }

  return 0;

}
