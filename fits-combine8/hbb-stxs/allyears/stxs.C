void stxs()
{
//=========Macro generated from canvas: stxs/stxs
//=========  (Tue Jun 25 14:53:56 2024) by ROOT version 6.12/07
   TCanvas *stxs = new TCanvas("stxs", "stxs",0,23,800,600);
   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   stxs->SetHighLightColor(2);
   stxs->Range(0,0,1,1);
   stxs->SetFillColor(0);
   stxs->SetBorderMode(0);
   stxs->SetBorderSize(2);
   stxs->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: pad1
   TPad *pad1 = new TPad("pad1", "pad1",0,0.4,0.6,1);
   pad1->Draw();
   pad1->cd();
   pad1->Range(-1.042169,-0.2045366,2.572289,3.886194);
   pad1->SetFillColor(0);
   pad1->SetBorderMode(0);
   pad1->SetBorderSize(2);
   pad1->SetLogy();
   pad1->SetLeftMargin(0.15);
   pad1->SetRightMargin(0.02);
   pad1->SetBottomMargin(0.05);
   pad1->SetFrameBorderMode(0);
   pad1->SetFrameBorderMode(0);
   
   TH1D *dummy1__1 = new TH1D("dummy1__1","dummy1",3,-0.5,2.5);
   dummy1__1->SetMinimum(1);
   dummy1__1->SetMaximum(3000);
   dummy1__1->SetEntries(3);
   dummy1__1->SetStats(0);
   dummy1__1->SetLineColor(0);
   dummy1__1->SetLineWidth(3);
   dummy1__1->GetXaxis()->SetTitle("p_{T}^{H} [GeV]");
   dummy1__1->GetXaxis()->SetBinLabel(1,"[300,450]");
   dummy1__1->GetXaxis()->SetBinLabel(2,"[450,650]");
   dummy1__1->GetXaxis()->SetBinLabel(3,"[650,#infty)");
   dummy1__1->GetXaxis()->CenterTitle(true);
   dummy1__1->GetXaxis()->SetLabelFont(42);
   dummy1__1->GetXaxis()->SetLabelSize(0);
   dummy1__1->GetXaxis()->SetTitleSize(0);
   dummy1__1->GetXaxis()->SetTitleOffset(1.5);
   dummy1__1->GetXaxis()->SetTitleFont(42);
   dummy1__1->GetYaxis()->SetTitle("#sigma_{obs} [fb]");
   dummy1__1->GetYaxis()->SetLabelFont(42);
   dummy1__1->GetYaxis()->SetLabelSize(0.05516841);
   dummy1__1->GetYaxis()->SetTitleSize(0.05516841);
   dummy1__1->GetYaxis()->SetTitleOffset(1.2);
   dummy1__1->GetYaxis()->SetTitleFont(42);
   dummy1__1->GetZaxis()->SetLabelFont(42);
   dummy1__1->GetZaxis()->SetLabelSize(0.035);
   dummy1__1->GetZaxis()->SetTitleSize(0.035);
   dummy1__1->GetZaxis()->SetTitleFont(42);
   dummy1__1->Draw("");
   
   Double_t Graph0_fx3001[3] = {
   0,
   1,
   2};
   Double_t Graph0_fy3001[3] = {
   88.24261,
   13.56206,
   1.781883};
   Double_t Graph0_felx3001[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fely3001[3] = {
   18.06141,
   2.781063,
   0.3782296};
   Double_t Graph0_fehx3001[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fehy3001[3] = {
   21.12193,
   3.100653,
   0.3896675};
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(3,Graph0_fx3001,Graph0_fy3001,Graph0_felx3001,Graph0_fehx3001,Graph0_fely3001,Graph0_fehy3001);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph3001 = new TH1F("Graph_Graph3001","Graph",100,-0.8,2.8);
   Graph_Graph3001->SetMinimum(1.263288);
   Graph_Graph3001->SetMaximum(120.1606);
   Graph_Graph3001->SetDirectory(0);
   Graph_Graph3001->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   Graph_Graph3001->SetLineColor(ci);
   Graph_Graph3001->GetXaxis()->SetLabelFont(42);
   Graph_Graph3001->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3001->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3001->GetXaxis()->SetTitleFont(42);
   Graph_Graph3001->GetYaxis()->SetLabelFont(42);
   Graph_Graph3001->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3001->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3001->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3001->GetYaxis()->SetTitleFont(42);
   Graph_Graph3001->GetZaxis()->SetLabelFont(42);
   Graph_Graph3001->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3001->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3001);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3002[3] = {
   0,
   1,
   2};
   Double_t Graph0_fy3002[3] = {
   88.24261,
   13.56206,
   1.781883};
   Double_t Graph0_felx3002[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fely3002[3] = {
   18.06141,
   2.781063,
   0.3782296};
   Double_t Graph0_fehx3002[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fehy3002[3] = {
   21.12193,
   3.100653,
   0.3896675};
   grae = new TGraphAsymmErrors(3,Graph0_fx3002,Graph0_fy3002,Graph0_felx3002,Graph0_fehx3002,Graph0_fely3002,Graph0_fehy3002);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph_Graph30013002 = new TH1F("Graph_Graph_Graph30013002","Graph",100,-0.8,2.8);
   Graph_Graph_Graph30013002->SetMinimum(1.263288);
   Graph_Graph_Graph30013002->SetMaximum(120.1606);
   Graph_Graph_Graph30013002->SetDirectory(0);
   Graph_Graph_Graph30013002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_Graph30013002->SetLineColor(ci);
   Graph_Graph_Graph30013002->GetXaxis()->SetLabelFont(42);
   Graph_Graph_Graph30013002->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30013002->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30013002->GetXaxis()->SetTitleFont(42);
   Graph_Graph_Graph30013002->GetYaxis()->SetLabelFont(42);
   Graph_Graph_Graph30013002->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30013002->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30013002->GetYaxis()->SetTitleOffset(0);
   Graph_Graph_Graph30013002->GetYaxis()->SetTitleFont(42);
   Graph_Graph_Graph30013002->GetZaxis()->SetLabelFont(42);
   Graph_Graph_Graph30013002->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30013002->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30013002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph_Graph30013002);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3003[3] = {
   0,
   1,
   2};
   Double_t Graph1_fy3003[3] = {
   -188.045,
   25.56448,
   4.486781};
   Double_t Graph1_felx3003[3] = {
   0,
   0,
   0};
   Double_t Graph1_fely3003[3] = {
   433.9772,
   26.55451,
   6.024546};
   Double_t Graph1_fehx3003[3] = {
   0,
   0,
   0};
   Double_t Graph1_fehy3003[3] = {
   445.9782,
   25.25255,
   5.4882};
   grae = new TGraphAsymmErrors(3,Graph1_fx3003,Graph1_fy3003,Graph1_felx3003,Graph1_fehx3003,Graph1_fely3003,Graph1_fehy3003);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3003 = new TH1F("Graph_Graph3003","Graph",100,0,2.2);
   Graph_Graph3003->SetMinimum(0.302531);
   Graph_Graph3003->SetMaximum(302.531);
   Graph_Graph3003->SetDirectory(0);
   Graph_Graph3003->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3003->SetLineColor(ci);
   Graph_Graph3003->GetXaxis()->SetLabelFont(42);
   Graph_Graph3003->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3003->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3003->GetXaxis()->SetTitleFont(42);
   Graph_Graph3003->GetYaxis()->SetLabelFont(42);
   Graph_Graph3003->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3003->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3003->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3003->GetYaxis()->SetTitleFont(42);
   Graph_Graph3003->GetZaxis()->SetLabelFont(42);
   Graph_Graph3003->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3003->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3003->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3003);
   
   grae->Draw("pe");
   TLatex *   tex = new TLatex(0.17,0.92,"CMS");
tex->SetNDC();
   tex->SetTextSize(0.05516841);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.26,0.92,"H(bb) STXS stage 1.2");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.05516841);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.2,0.82,"ggF");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.05516841);
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.44,0.65,0.82,0.87,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.05516841);
   leg->SetLineColor(0);
   leg->SetLineStyle(0);
   leg->SetLineWidth(0);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("histo","Observed (stat #oplus syst)","pe");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(3);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph0","ggF (HJMINLO)","f");
   entry->SetFillColor(4);
   entry->SetFillStyle(3003);
   entry->SetLineColor(4);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph","VBF (POWHEG+HC)","f");
   entry->SetFillColor(94);
   entry->SetFillStyle(3003);
   entry->SetLineColor(94);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   pad1->Modified();
   stxs->cd();
  
// ------------>Primitives in pad: pad2
   TPad *pad2 = new TPad("pad2", "pad2",0,0,0.6,0.4);
   pad2->Draw();
   pad2->cd();
   pad2->Range(-1.042169,-14.00013,2.572289,16.0003);
   pad2->SetFillColor(0);
   pad2->SetBorderMode(0);
   pad2->SetBorderSize(2);
   pad2->SetLeftMargin(0.15);
   pad2->SetRightMargin(0.02);
   pad2->SetTopMargin(1e-05);
   pad2->SetBottomMargin(0.3);
   pad2->SetFrameBorderMode(0);
   pad2->SetFrameBorderMode(0);
   
   TH1D *dummy3__2 = new TH1D("dummy3__2","dummy1",3,-0.5,2.5);
   dummy3__2->SetMinimum(-5);
   dummy3__2->SetMaximum(16);
   dummy3__2->SetStats(0);
   dummy3__2->SetLineColor(0);
   dummy3__2->SetLineWidth(3);
   dummy3__2->GetXaxis()->SetTitle("p_{T}^{H} [GeV]");
   dummy3__2->GetXaxis()->SetBinLabel(1,"[300,450]");
   dummy3__2->GetXaxis()->SetBinLabel(2,"[450,650]");
   dummy3__2->GetXaxis()->SetBinLabel(3,"[650,#infty)");
   dummy3__2->GetXaxis()->CenterTitle(true);
   dummy3__2->GetXaxis()->SetLabelFont(42);
   dummy3__2->GetXaxis()->SetLabelSize(0.1075784);
   dummy3__2->GetXaxis()->SetTitleSize(0.08275262);
   dummy3__2->GetXaxis()->SetTitleOffset(1.5);
   dummy3__2->GetXaxis()->SetTitleFont(42);
   dummy3__2->GetYaxis()->SetTitle("#sigma_{obs} / #sigma_{SM}");
   dummy3__2->GetYaxis()->SetLabelFont(42);
   dummy3__2->GetYaxis()->SetLabelSize(0.08275262);
   dummy3__2->GetYaxis()->SetTitleSize(0.08275262);
   dummy3__2->GetYaxis()->SetTitleOffset(0.8);
   dummy3__2->GetYaxis()->SetTitleFont(42);
   dummy3__2->GetZaxis()->SetLabelFont(42);
   dummy3__2->GetZaxis()->SetLabelSize(0.035);
   dummy3__2->GetZaxis()->SetTitleSize(0.035);
   dummy3__2->GetZaxis()->SetTitleFont(42);
   dummy3__2->Draw("");
   
   Double_t Graph0_fx3004[3] = {
   0,
   1,
   2};
   Double_t Graph0_fy3004[3] = {
   1,
   1,
   1};
   Double_t Graph0_felx3004[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fely3004[3] = {
   0.204679,
   0.205062,
   0.212264};
   Double_t Graph0_fehx3004[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fehy3004[3] = {
   0.239362,
   0.228627,
   0.218683};
   grae = new TGraphAsymmErrors(3,Graph0_fx3004,Graph0_fy3004,Graph0_felx3004,Graph0_fehx3004,Graph0_fely3004,Graph0_fehy3004);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph3004 = new TH1F("Graph_Graph3004","Graph",100,-0.8,2.8);
   Graph_Graph3004->SetMinimum(0.7425734);
   Graph_Graph3004->SetMaximum(1.284525);
   Graph_Graph3004->SetDirectory(0);
   Graph_Graph3004->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3004->SetLineColor(ci);
   Graph_Graph3004->GetXaxis()->SetLabelFont(42);
   Graph_Graph3004->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3004->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3004->GetXaxis()->SetTitleFont(42);
   Graph_Graph3004->GetYaxis()->SetLabelFont(42);
   Graph_Graph3004->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3004->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3004->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3004->GetYaxis()->SetTitleFont(42);
   Graph_Graph3004->GetZaxis()->SetLabelFont(42);
   Graph_Graph3004->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3004->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3004->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3004);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3005[3] = {
   0,
   1,
   2};
   Double_t Graph0_fy3005[3] = {
   1,
   1,
   1};
   Double_t Graph0_felx3005[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fely3005[3] = {
   0.204679,
   0.205062,
   0.212264};
   Double_t Graph0_fehx3005[3] = {
   0.5,
   0.5,
   0.5};
   Double_t Graph0_fehy3005[3] = {
   0.239362,
   0.228627,
   0.218683};
   grae = new TGraphAsymmErrors(3,Graph0_fx3005,Graph0_fy3005,Graph0_felx3005,Graph0_fehx3005,Graph0_fely3005,Graph0_fehy3005);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   
   TH1F *Graph_Graph_Graph30043005 = new TH1F("Graph_Graph_Graph30043005","Graph",100,-0.8,2.8);
   Graph_Graph_Graph30043005->SetMinimum(0.7425734);
   Graph_Graph_Graph30043005->SetMaximum(1.284525);
   Graph_Graph_Graph30043005->SetDirectory(0);
   Graph_Graph_Graph30043005->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_Graph30043005->SetLineColor(ci);
   Graph_Graph_Graph30043005->GetXaxis()->SetLabelFont(42);
   Graph_Graph_Graph30043005->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30043005->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30043005->GetXaxis()->SetTitleFont(42);
   Graph_Graph_Graph30043005->GetYaxis()->SetLabelFont(42);
   Graph_Graph_Graph30043005->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30043005->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30043005->GetYaxis()->SetTitleOffset(0);
   Graph_Graph_Graph30043005->GetYaxis()->SetTitleFont(42);
   Graph_Graph_Graph30043005->GetZaxis()->SetLabelFont(42);
   Graph_Graph_Graph30043005->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30043005->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30043005->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph_Graph30043005);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3006[3] = {
   0,
   1,
   2};
   Double_t Graph1_fy3006[3] = {
   -2.131,
   1.885,
   2.518};
   Double_t Graph1_felx3006[3] = {
   0,
   0,
   0};
   Double_t Graph1_fely3006[3] = {
   4.918,
   1.958,
   3.381};
   Double_t Graph1_fehx3006[3] = {
   0,
   0,
   0};
   Double_t Graph1_fehy3006[3] = {
   5.054,
   1.862,
   3.08};
   grae = new TGraphAsymmErrors(3,Graph1_fx3006,Graph1_fy3006,Graph1_felx3006,Graph1_fehx3006,Graph1_fely3006,Graph1_fehy3006);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3006 = new TH1F("Graph_Graph3006","Graph",100,0,2.2);
   Graph_Graph3006->SetMinimum(-8.3137);
   Graph_Graph3006->SetMaximum(6.8627);
   Graph_Graph3006->SetDirectory(0);
   Graph_Graph3006->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3006->SetLineColor(ci);
   Graph_Graph3006->GetXaxis()->SetLabelFont(42);
   Graph_Graph3006->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3006->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3006->GetXaxis()->SetTitleFont(42);
   Graph_Graph3006->GetYaxis()->SetLabelFont(42);
   Graph_Graph3006->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3006->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3006->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3006->GetYaxis()->SetTitleFont(42);
   Graph_Graph3006->GetZaxis()->SetLabelFont(42);
   Graph_Graph3006->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3006->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3006->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3006);
   
   grae->Draw("pe");
   pad2->Modified();
   stxs->cd();
  
// ------------>Primitives in pad: pad3
   TPad *pad3 = new TPad("pad3", "pad3",0.6,0.4,1,1);
   pad3->Draw();
   pad3->cd();
   pad3->Range(2.451807,-0.2045366,4.861446,3.886194);
   pad3->SetFillColor(0);
   pad3->SetBorderMode(0);
   pad3->SetBorderSize(2);
   pad3->SetLogy();
   pad3->SetLeftMargin(0.02);
   pad3->SetRightMargin(0.15);
   pad3->SetBottomMargin(0.05);
   pad3->SetFrameBorderMode(0);
   pad3->SetFrameBorderMode(0);
   
   TH1D *dummy2__3 = new TH1D("dummy2__3","dummy2",2,2.5,4.5);
   dummy2__3->SetMinimum(1);
   dummy2__3->SetMaximum(3000);
   dummy2__3->SetEntries(2);
   dummy2__3->SetStats(0);
   dummy2__3->SetLineColor(0);
   dummy2__3->SetLineWidth(3);
   dummy2__3->GetXaxis()->SetTitle("m_{jj}^{gen} [GeV]");
   dummy2__3->GetXaxis()->SetBinLabel(1,"[1000,1500]");
   dummy2__3->GetXaxis()->SetBinLabel(2,"[1500,#infty)");
   dummy2__3->GetXaxis()->CenterTitle(true);
   dummy2__3->GetXaxis()->SetLabelFont(42);
   dummy2__3->GetXaxis()->SetLabelSize(0);
   dummy2__3->GetXaxis()->SetTitleSize(0);
   dummy2__3->GetXaxis()->SetTitleOffset(1.5);
   dummy2__3->GetXaxis()->SetTitleFont(42);
   dummy2__3->GetYaxis()->SetLabelFont(42);
   dummy2__3->GetYaxis()->SetLabelSize(0);
   dummy2__3->GetYaxis()->SetTitleSize(0);
   dummy2__3->GetYaxis()->SetTitleOffset(0);
   dummy2__3->GetYaxis()->SetTitleFont(42);
   dummy2__3->GetZaxis()->SetLabelFont(42);
   dummy2__3->GetZaxis()->SetLabelSize(0.035);
   dummy2__3->GetZaxis()->SetTitleSize(0.035);
   dummy2__3->GetZaxis()->SetTitleFont(42);
   dummy2__3->Draw("");
   
   Double_t Graph0_fx3007[2] = {
   3,
   4};
   Double_t Graph0_fy3007[2] = {
   30.67989,
   40.70797};
   Double_t Graph0_felx3007[2] = {
   0.5,
   0.5};
   Double_t Graph0_fely3007[2] = {
   1.659813,
   3.133862};
   Double_t Graph0_fehx3007[2] = {
   0.5,
   0.5};
   Double_t Graph0_fehy3007[2] = {
   1.354793,
   2.424485};
   grae = new TGraphAsymmErrors(2,Graph0_fx3007,Graph0_fy3007,Graph0_felx3007,Graph0_fehx3007,Graph0_fely3007,Graph0_fehy3007);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph3007 = new TH1F("Graph_Graph3007","Graph",100,2.3,4.7);
   Graph_Graph3007->SetMinimum(27.60884);
   Graph_Graph3007->SetMaximum(44.54369);
   Graph_Graph3007->SetDirectory(0);
   Graph_Graph3007->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3007->SetLineColor(ci);
   Graph_Graph3007->GetXaxis()->SetLabelFont(42);
   Graph_Graph3007->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3007->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3007->GetXaxis()->SetTitleFont(42);
   Graph_Graph3007->GetYaxis()->SetLabelFont(42);
   Graph_Graph3007->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3007->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3007->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3007->GetYaxis()->SetTitleFont(42);
   Graph_Graph3007->GetZaxis()->SetLabelFont(42);
   Graph_Graph3007->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3007->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3007->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3007);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3008[2] = {
   3,
   4};
   Double_t Graph0_fy3008[2] = {
   30.67989,
   40.70797};
   Double_t Graph0_felx3008[2] = {
   0.5,
   0.5};
   Double_t Graph0_fely3008[2] = {
   1.659813,
   3.133862};
   Double_t Graph0_fehx3008[2] = {
   0.5,
   0.5};
   Double_t Graph0_fehy3008[2] = {
   1.354793,
   2.424485};
   grae = new TGraphAsymmErrors(2,Graph0_fx3008,Graph0_fy3008,Graph0_felx3008,Graph0_fehx3008,Graph0_fely3008,Graph0_fehy3008);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph_Graph30073008 = new TH1F("Graph_Graph_Graph30073008","Graph",100,2.3,4.7);
   Graph_Graph_Graph30073008->SetMinimum(27.60884);
   Graph_Graph_Graph30073008->SetMaximum(44.54369);
   Graph_Graph_Graph30073008->SetDirectory(0);
   Graph_Graph_Graph30073008->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_Graph30073008->SetLineColor(ci);
   Graph_Graph_Graph30073008->GetXaxis()->SetLabelFont(42);
   Graph_Graph_Graph30073008->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30073008->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30073008->GetXaxis()->SetTitleFont(42);
   Graph_Graph_Graph30073008->GetYaxis()->SetLabelFont(42);
   Graph_Graph_Graph30073008->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30073008->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30073008->GetYaxis()->SetTitleOffset(0);
   Graph_Graph_Graph30073008->GetYaxis()->SetTitleFont(42);
   Graph_Graph_Graph30073008->GetZaxis()->SetLabelFont(42);
   Graph_Graph_Graph30073008->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30073008->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30073008->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph_Graph30073008);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3009[2] = {
   3,
   4};
   Double_t Graph1_fy3009[2] = {
   236.542,
   124.4443};
   Double_t Graph1_felx3009[2] = {
   0,
   0};
   Double_t Graph1_fely3009[2] = {
   174.9368,
   62.3239};
   Double_t Graph1_fehx3009[2] = {
   0,
   0};
   Double_t Graph1_fehy3009[2] = {
   170.12,
   59.35222};
   grae = new TGraphAsymmErrors(2,Graph1_fx3009,Graph1_fy3009,Graph1_felx3009,Graph1_fehx3009,Graph1_fely3009,Graph1_fehy3009);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3009 = new TH1F("Graph_Graph3009","Graph",100,2.9,4.1);
   Graph_Graph3009->SetMinimum(27.09955);
   Graph_Graph3009->SetMaximum(441.1677);
   Graph_Graph3009->SetDirectory(0);
   Graph_Graph3009->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3009->SetLineColor(ci);
   Graph_Graph3009->GetXaxis()->SetLabelFont(42);
   Graph_Graph3009->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3009->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3009->GetXaxis()->SetTitleFont(42);
   Graph_Graph3009->GetYaxis()->SetLabelFont(42);
   Graph_Graph3009->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3009->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3009->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3009->GetYaxis()->SetTitleFont(42);
   Graph_Graph3009->GetZaxis()->SetLabelFont(42);
   Graph_Graph3009->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3009->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3009->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3009);
   
   grae->Draw("p");
      tex = new TLatex(0.46,0.92,"138 fb^{-1} (13 TeV)");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.05516841);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.08,0.82,"VBF");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.05516841);
   tex->SetLineWidth(2);
   tex->Draw();
   pad3->Modified();
   stxs->cd();
  
// ------------>Primitives in pad: pad4
   TPad *pad4 = new TPad("pad4", "pad4",0.6,0,1,0.4);
   pad4->Draw();
   pad4->cd();
   pad4->Range(2.451807,-14.00013,4.861446,16.0003);
   pad4->SetFillColor(0);
   pad4->SetBorderMode(0);
   pad4->SetBorderSize(2);
   pad4->SetLeftMargin(0.02);
   pad4->SetRightMargin(0.15);
   pad4->SetTopMargin(1e-05);
   pad4->SetBottomMargin(0.3);
   pad4->SetFrameBorderMode(0);
   pad4->SetFrameBorderMode(0);
   
   TH1D *dummy4__4 = new TH1D("dummy4__4","dummy2",2,2.5,4.5);
   dummy4__4->SetMinimum(-5);
   dummy4__4->SetMaximum(16);
   dummy4__4->SetStats(0);
   dummy4__4->SetLineColor(0);
   dummy4__4->SetLineWidth(3);
   dummy4__4->GetXaxis()->SetTitle("m_{jj}^{gen} [GeV]");
   dummy4__4->GetXaxis()->SetBinLabel(1,"[1000,1500]");
   dummy4__4->GetXaxis()->SetBinLabel(2,"[1500,#infty)");
   dummy4__4->GetXaxis()->CenterTitle(true);
   dummy4__4->GetXaxis()->SetLabelFont(42);
   dummy4__4->GetXaxis()->SetLabelSize(0.1075784);
   dummy4__4->GetXaxis()->SetTitleSize(0.08275262);
   dummy4__4->GetXaxis()->SetTitleOffset(1.5);
   dummy4__4->GetXaxis()->SetTitleFont(42);
   dummy4__4->GetYaxis()->SetLabelFont(42);
   dummy4__4->GetYaxis()->SetLabelSize(0);
   dummy4__4->GetYaxis()->SetTitleSize(0);
   dummy4__4->GetYaxis()->SetTitleOffset(0);
   dummy4__4->GetYaxis()->SetTitleFont(42);
   dummy4__4->GetZaxis()->SetLabelFont(42);
   dummy4__4->GetZaxis()->SetLabelSize(0.035);
   dummy4__4->GetZaxis()->SetTitleSize(0.035);
   dummy4__4->GetZaxis()->SetTitleFont(42);
   dummy4__4->Draw("");
   
   Double_t Graph0_fx3010[2] = {
   3,
   4};
   Double_t Graph0_fy3010[2] = {
   1,
   1};
   Double_t Graph0_felx3010[2] = {
   0.5,
   0.5};
   Double_t Graph0_fely3010[2] = {
   0.054101,
   0.076984};
   Double_t Graph0_fehx3010[2] = {
   0.5,
   0.5};
   Double_t Graph0_fehy3010[2] = {
   0.044159,
   0.059558};
   grae = new TGraphAsymmErrors(2,Graph0_fx3010,Graph0_fy3010,Graph0_felx3010,Graph0_fehx3010,Graph0_fely3010,Graph0_fehy3010);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph3010 = new TH1F("Graph_Graph3010","Graph",100,2.3,4.7);
   Graph_Graph3010->SetMinimum(0.9093618);
   Graph_Graph3010->SetMaximum(1.073212);
   Graph_Graph3010->SetDirectory(0);
   Graph_Graph3010->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3010->SetLineColor(ci);
   Graph_Graph3010->GetXaxis()->SetLabelFont(42);
   Graph_Graph3010->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3010->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3010->GetXaxis()->SetTitleFont(42);
   Graph_Graph3010->GetYaxis()->SetLabelFont(42);
   Graph_Graph3010->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3010->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3010->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3010->GetYaxis()->SetTitleFont(42);
   Graph_Graph3010->GetZaxis()->SetLabelFont(42);
   Graph_Graph3010->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3010->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3010->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3010);
   
   grae->Draw("2");
   
   Double_t Graph0_fx3011[2] = {
   3,
   4};
   Double_t Graph0_fy3011[2] = {
   1,
   1};
   Double_t Graph0_felx3011[2] = {
   0.5,
   0.5};
   Double_t Graph0_fely3011[2] = {
   0.054101,
   0.076984};
   Double_t Graph0_fehx3011[2] = {
   0.5,
   0.5};
   Double_t Graph0_fehy3011[2] = {
   0.044159,
   0.059558};
   grae = new TGraphAsymmErrors(2,Graph0_fx3011,Graph0_fy3011,Graph0_felx3011,Graph0_fehx3011,Graph0_fely3011,Graph0_fehy3011);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(3003);
   grae->SetLineColor(94);
   grae->SetMarkerColor(94);
   
   TH1F *Graph_Graph_Graph30103011 = new TH1F("Graph_Graph_Graph30103011","Graph",100,2.3,4.7);
   Graph_Graph_Graph30103011->SetMinimum(0.9093618);
   Graph_Graph_Graph30103011->SetMaximum(1.073212);
   Graph_Graph_Graph30103011->SetDirectory(0);
   Graph_Graph_Graph30103011->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_Graph30103011->SetLineColor(ci);
   Graph_Graph_Graph30103011->GetXaxis()->SetLabelFont(42);
   Graph_Graph_Graph30103011->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30103011->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30103011->GetXaxis()->SetTitleFont(42);
   Graph_Graph_Graph30103011->GetYaxis()->SetLabelFont(42);
   Graph_Graph_Graph30103011->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30103011->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30103011->GetYaxis()->SetTitleOffset(0);
   Graph_Graph_Graph30103011->GetYaxis()->SetTitleFont(42);
   Graph_Graph_Graph30103011->GetZaxis()->SetLabelFont(42);
   Graph_Graph_Graph30103011->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph30103011->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph30103011->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph_Graph30103011);
   
   grae->Draw("pe");
   
   Double_t Graph1_fx3012[2] = {
   3,
   4};
   Double_t Graph1_fy3012[2] = {
   7.707,
   3.057};
   Double_t Graph1_felx3012[2] = {
   0,
   0};
   Double_t Graph1_fely3012[2] = {
   6.422,
   1.661};
   Double_t Graph1_fehx3012[2] = {
   0,
   0};
   Double_t Graph1_fehy3012[2] = {
   6.035,
   1.49};
   grae = new TGraphAsymmErrors(2,Graph1_fx3012,Graph1_fy3012,Graph1_felx3012,Graph1_fehx3012,Graph1_fely3012,Graph1_fehy3012);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3012 = new TH1F("Graph_Graph3012","Graph",100,2.9,4.1);
   Graph_Graph3012->SetMinimum(0.0393);
   Graph_Graph3012->SetMaximum(14.9877);
   Graph_Graph3012->SetDirectory(0);
   Graph_Graph3012->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3012->SetLineColor(ci);
   Graph_Graph3012->GetXaxis()->SetLabelFont(42);
   Graph_Graph3012->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3012->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3012->GetXaxis()->SetTitleFont(42);
   Graph_Graph3012->GetYaxis()->SetLabelFont(42);
   Graph_Graph3012->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3012->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3012->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3012->GetYaxis()->SetTitleFont(42);
   Graph_Graph3012->GetZaxis()->SetLabelFont(42);
   Graph_Graph3012->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3012->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3012->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3012);
   
   grae->Draw("p");
   pad4->Modified();
   stxs->cd();
   stxs->Modified();
   stxs->cd();
   stxs->SetSelected(stxs);
}
