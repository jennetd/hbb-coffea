vector<string> systnames = {
  "TFres",
  "TTQ",
  "AllExp",
  "MCStat",
  "DDB",
  "JMSJMR",
  "JESJER",
  "qcdparam",
  "Other",
  "AllTh",
  "ggFtheory",
  "VBFtheory",
  "VHttHtheory",
  "Vtheory"
};

vector<double> read_from_graph(string poi, string filestring){
  
  TFile *f = new TFile(("higgsCombine"+filestring+".MultiDimFit.mH125.root").c_str());  
  
  // Variables                                                                                                                                         
  float x = 0;
  float y = 0;
  
  TTree* t = (TTree*)f->Get("limit");
  t->SetBranchAddress(poi.c_str(),&x);
  t->SetBranchAddress("deltaNLL",&y);
  int npoints = t->GetEntries();
  
  double r_pos[npoints-1];
  double dnll_pos[npoints-1];

  double r_neg[npoints-1];
  double dnll_neg[npoints-1];

  t->GetEntry(0);
  double center = x;
  
  for(int i=0; i<npoints; i++){
    
    t->GetEntry(i);
    
    if( i>0 ){
      if( x > center ){
	r_pos[i-1] = x;
	dnll_pos[i-1] = 2*y;

	r_neg[i-1] = 0;
	dnll_neg[i-1] = 0;
      }
      else{
	r_pos[i-1] = 0;
	dnll_pos[i-1] = 0;

	r_neg[i-1] = x;
	dnll_neg[i-1] = 2*y;
      }
    }

  }
  
  TGraph* gpos = new TGraph(npoints-1,dnll_pos,r_pos);
  TGraph* gneg = new TGraph(npoints-1,dnll_neg,r_neg);
  
  vector<double> unc;
  unc.push_back(gpos->Eval(center) - center);
  unc.push_back(gneg->Eval(center) - center);
  
  return unc;
}

vector<double> read_from_tree(string poi, string filestring){

  TFile *f = new TFile(("higgsCombine"+filestring+".MultiDimFit.mH125.root").c_str());

  // Variables                                                                                                                                               
  float x = 0;
  float y = 0;

  TTree* t = (TTree*)f->Get("limit");
  t->SetBranchAddress(poi.c_str(),&x);
  int npoints = t->GetEntries();

  vector<double> uncs = {1,1};

  t->GetEntry(0);
  double center = x;
  for(int i=0; i<npoints; i++){

    t->GetEntry(i);

    if(x > center) uncs.at(1) = x - center;
    if(x < center) uncs.at(0) = x - center;

  }
  return uncs;

}

vector<double> read_from_file(string poi, string filestring){

  vector<double> vals;

  string logfile = filestring;                                                                                                       
  ifstream infile;                                                                                                                   
  string line;                                                                                                                       
  string theline;                                                                                                                    

  infile.open(logfile.c_str());                                                                                                      
  while(getline(infile, line)) {                                                                                                     
    if (line.find(poi+" :", 0) != string::npos) {                                                                                    
      theline = line;                                                                                                                
    }                                                                                                                                
  }                                                                                                                                  
                                                                                                                                     
  vector<string> words;                                                                                                              
                                                                                                                                     
  istringstream iss(theline);                                                                                                        
  for (std::string s; iss >> s; )                                                                                                    
    words.push_back(s);                                                                                                              

  //  double center = stod(words.at(2));                                                                                          
  //  vals.push_back(center);     
  
  int splitpoint = words.at(3).find('/');                                                                                  
  
  double down_unc = stod(words.at(3).substr(0,splitpoint));                                                                           
  vals.push_back(down_unc);                                                                                                           
                                                                                                                                      
  double up_unc = stod(words.at(3).substr(splitpoint+1,string::npos));                                                                
  vals.push_back(up_unc);                                                                                                             
                                                                                                                                      
  return vals;
}

map<string,vector<double>> get_table(string poi){

  map<string,vector<double>> table; 

  cout << "----------- " << poi << endl;

  vector<double> unc_total = read_from_graph(poi,"_Total");
  table["Total"] = unc_total;

  vector<double> unc_allstat = read_from_graph(poi, "_"+poi+"AllStat");
  table["AllStat"] = unc_allstat;

  vector<double> unc_sigextr = read_from_graph(poi, "_"+poi+"SigExtr");
  table["SigExtr"] = unc_sigextr;

  for(int i=0; i<systnames.size(); i++){

    vector<double> unc = read_from_graph(poi, "_"+poi+systnames.at(i));

    double delta_mu_m = sqrt(unc_total[0]*unc_total[0] - unc[0]*unc[0]);
    double delta_mu_p = sqrt(unc_total[1]*unc_total[1] - unc[1]*unc[1]);

    unc.at(0) = delta_mu_m;
    unc.at(1) = delta_mu_p;
    table[systnames.at(i)] = unc;

  }

  return table;

}
 
void syst_table(){

  vector<string> allsystnames = {"Total","AllStat","SigExtr"};
  allsystnames.insert(allsystnames.end(), systnames.begin(), systnames.end());

  map<string,vector<double>> vbftable = get_table("rVBF");
  map<string,vector<double>> ggftable = get_table("rggF");

  map<string,string> systtitles;
  systtitles["Total"] = "Total";
  systtitles["AllStat"] = "Statistical";
  systtitles["SigExtr"] = "Signal extraction";
  systtitles["TFres"] = "QCD estimation ($F_{res}$)";
  systtitles["TTQ"] = "$t\bar{t}$ normalization & efficiency";
  systtitles["AllExp"] = "Experimental";
  systtitles["MCStat"] = "Size of simulated samples";
  systtitles["DDB"] = "DDB tag efficiency";
  systtitles["JMSJMR"] = "Jet mass scale & resolution";
  systtitles["JESJER"] = "Jet energy scale & resolution";
  systtitles["qcdparam"] = "QCD estimation ($F_{P/F}$)";
  systtitles["Other"] = "Other";
  systtitles["AllTh"] = "Theoretical";
  systtitles["ggFtheory"] = "ggF modeling";
  systtitles["VBFtheory"] = "VBF modeling";
  systtitles["VHttHtheory"] = "Other Higgs modeling";
  systtitles["Vtheory"]= "V+jets modeling";


  for(int i=0; i<allsystnames.size(); i++){
    string sys = allsystnames.at(i);
    cout << std::setprecision(3) << systtitles[sys] << " &\t" << vbftable[sys].at(0) << " &\t" << vbftable[sys].at(1) << " &\t" << ggftable[sys].at(0) << " &\t" << ggftable[sys].at(1) << endl;
  }


  return;
}
