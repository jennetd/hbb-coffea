void mu_bybin()
{
//=========Macro generated from canvas: c/c
//=========  (Thu May 30 08:09:07 2024) by ROOT version 6.12/07
   TCanvas *c = new TCanvas("c", "c",235,1103,800,800);
   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   c->SetHighLightColor(2);
   c->Range(0,0,1,1);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: pad1
   TPad *pad1 = new TPad("pad1", "pad1",0,0.66,1,1);
   pad1->Draw();
   pad1->cd();
   pad1->Range(-30,-0.3333333,20,3);
   pad1->SetFillColor(0);
   pad1->SetBorderMode(0);
   pad1->SetBorderSize(2);
   pad1->SetLeftMargin(0.3);
   pad1->SetTopMargin(0.15);
   pad1->SetBottomMargin(0.25);
   pad1->SetFrameBorderMode(0);
   pad1->SetFrameBorderMode(0);
   
   TH1D *dummy1__1 = new TH1D("dummy1__1","dummy1",2,0.5,2.5);
   dummy1__1->SetBinContent(1,-15);
   dummy1__1->SetBinContent(2,-15);
   dummy1__1->SetBinError(1,15);
   dummy1__1->SetBinError(2,15);
   dummy1__1->SetMinimum(-15);
   dummy1__1->SetMaximum(15);
   dummy1__1->SetEntries(2);
   dummy1__1->SetStats(0);
   dummy1__1->SetLineColor(0);
   dummy1__1->GetXaxis()->SetTitle("VBF category");
   dummy1__1->GetXaxis()->SetBinLabel(1,"1000 < m_{jj} < 2000 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(2,"m_{jj} > 2000 GeV");
   dummy1__1->GetXaxis()->SetLabelFont(42);
   dummy1__1->GetXaxis()->SetLabelSize(0.1);
   dummy1__1->GetXaxis()->SetTitleSize(0.1);
   dummy1__1->GetXaxis()->SetTitleOffset(1.5);
   dummy1__1->GetXaxis()->SetTitleFont(42);
   dummy1__1->GetYaxis()->SetTitle("#mu_{VBF}");
   dummy1__1->GetYaxis()->SetLabelFont(42);
   dummy1__1->GetYaxis()->SetLabelSize(0.1);
   dummy1__1->GetYaxis()->SetTitleSize(0.1);
   dummy1__1->GetYaxis()->SetTitleOffset(0.75);
   dummy1__1->GetYaxis()->SetTitleFont(42);
   dummy1__1->GetZaxis()->SetLabelFont(42);
   dummy1__1->GetZaxis()->SetLabelSize(0.035);
   dummy1__1->GetZaxis()->SetTitleSize(0.035);
   dummy1__1->GetZaxis()->SetTitleFont(42);
   dummy1__1->Draw("hbar");
   
   Double_t Graph0_fx1[2] = {
   1,
   1};
   Double_t Graph0_fy1[2] = {
   -100,
   100};
   TGraph *graph = new TGraph(2,Graph0_fx1,Graph0_fy1);
   graph->SetName("Graph0");
   graph->SetTitle("Graph");
   graph->SetFillStyle(1000);
   graph->SetLineStyle(3);
   graph->SetLineWidth(3);
   
   TH1F *Graph_Graph1 = new TH1F("Graph_Graph1","Graph",100,0.9,2.1);
   Graph_Graph1->SetMinimum(-120);
   Graph_Graph1->SetMaximum(120);
   Graph_Graph1->SetDirectory(0);
   Graph_Graph1->SetStats(0);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#000099");
   Graph_Graph1->SetLineColor(ci);
   Graph_Graph1->GetXaxis()->SetLabelFont(42);
   Graph_Graph1->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph1->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph1->GetXaxis()->SetTitleFont(42);
   Graph_Graph1->GetYaxis()->SetLabelFont(42);
   Graph_Graph1->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph1->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph1->GetYaxis()->SetTitleOffset(0);
   Graph_Graph1->GetYaxis()->SetTitleFont(42);
   Graph_Graph1->GetZaxis()->SetLabelFont(42);
   Graph_Graph1->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph1->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph1->GetZaxis()->SetTitleFont(42);
   graph->SetHistogram(Graph_Graph1);
   
   graph->Draw("l");
   
   Double_t Graph1_fx3001[2] = {
   3.26,
   6.763};
   Double_t Graph1_fy3001[2] = {
   0,
   0};
   Double_t Graph1_felx3001[2] = {
   0,
   0};
   Double_t Graph1_fely3001[2] = {
   -100,
   100};
   Double_t Graph1_fehx3001[2] = {
   0,
   0};
   Double_t Graph1_fehy3001[2] = {
   -100,
   100};
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(2,Graph1_fx3001,Graph1_fy3001,Graph1_felx3001,Graph1_fehx3001,Graph1_fely3001,Graph1_fehy3001);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   grae->SetLineWidth(3);
   
   TH1F *Graph_Graph3001 = new TH1F("Graph_Graph3001","Graph",100,2.9097,7.1133);
   Graph_Graph3001->SetMinimum(-120);
   Graph_Graph3001->SetMaximum(120);
   Graph_Graph3001->SetDirectory(0);
   Graph_Graph3001->SetStats(0);

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
   
   grae->Draw("3");
   
   Double_t Graph2_fx3002[2] = {
   4.885,
   4.885};
   Double_t Graph2_fy3002[2] = {
   -100,
   100};
   Double_t Graph2_felx3002[2] = {
   1.625,
   1.625};
   Double_t Graph2_fely3002[2] = {
   0,
   0};
   Double_t Graph2_fehx3002[2] = {
   1.878,
   1.878};
   Double_t Graph2_fehy3002[2] = {
   0,
   0};
   grae = new TGraphAsymmErrors(2,Graph2_fx3002,Graph2_fy3002,Graph2_felx3002,Graph2_fehx3002,Graph2_fely3002,Graph2_fehy3002);
   grae->SetName("Graph2");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   grae->SetLineWidth(3);
   
   TH1F *Graph_Graph3002 = new TH1F("Graph_Graph3002","Graph",100,2.9097,7.1133);
   Graph_Graph3002->SetMinimum(-120);
   Graph_Graph3002->SetMaximum(120);
   Graph_Graph3002->SetDirectory(0);
   Graph_Graph3002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3002->SetLineColor(ci);
   Graph_Graph3002->GetXaxis()->SetLabelFont(42);
   Graph_Graph3002->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3002->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3002->GetXaxis()->SetTitleFont(42);
   Graph_Graph3002->GetYaxis()->SetLabelFont(42);
   Graph_Graph3002->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3002->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3002->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3002->GetYaxis()->SetTitleFont(42);
   Graph_Graph3002->GetZaxis()->SetLabelFont(42);
   Graph_Graph3002->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3002->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3002);
   
   grae->Draw("l");
   
   Double_t Graph3_fx3003[2] = {
   6.203,
   4.223};
   Double_t Graph3_fy3003[2] = {
   1,
   2};
   Double_t Graph3_felx3003[2] = {
   2.551,
   1.638};
   Double_t Graph3_fely3003[2] = {
   0,
   0};
   Double_t Graph3_fehx3003[2] = {
   2.636,
   1.718};
   Double_t Graph3_fehy3003[2] = {
   0,
   0};
   grae = new TGraphAsymmErrors(2,Graph3_fx3003,Graph3_fy3003,Graph3_felx3003,Graph3_fehx3003,Graph3_fely3003,Graph3_fehy3003);
   grae->SetName("Graph3");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineColor(94);
   grae->SetLineWidth(20);
   grae->SetMarkerSize(0);
   
   TH1F *Graph_Graph3003 = new TH1F("Graph_Graph3003","Graph",100,1.9596,9.4644);
   Graph_Graph3003->SetMinimum(0.9);
   Graph_Graph3003->SetMaximum(2.1);
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
   
   grae->Draw("p");
   
   Double_t Graph4_fx3004[2] = {
   6.206,
   4.223};
   Double_t Graph4_fy3004[2] = {
   1,
   2};
   Double_t Graph4_felx3004[2] = {
   2.671,
   1.701};
   Double_t Graph4_fely3004[2] = {
   0,
   0};
   Double_t Graph4_fehx3004[2] = {
   3.028,
   1.996};
   Double_t Graph4_fehy3004[2] = {
   0,
   0};
   grae = new TGraphAsymmErrors(2,Graph4_fx3004,Graph4_fy3004,Graph4_felx3004,Graph4_fehx3004,Graph4_fely3004,Graph4_fehy3004);
   grae->SetName("Graph4");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3004 = new TH1F("Graph_Graph3004","Graph",100,1.8508,9.9052);
   Graph_Graph3004->SetMinimum(0.9);
   Graph_Graph3004->SetMaximum(2.1);
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
   
   grae->Draw("p");
   TLatex *   tex = new TLatex(0.68,0.88,"138 fb^{-1} (13 TeV)");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.1);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.3,0.88,"#bf{CMS}");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.1);
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.33,0.35,0.6,0.82,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetLineColor(0);
   leg->SetLineStyle(0);
   leg->SetLineWidth(0);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("Graph","Observed","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph","#pm1#sigma (stat #oplus syst)","l");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(3);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph","#pm1#sigma (stat)","f");
   entry->SetFillColor(94);
   entry->SetFillStyle(1000);
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph","Combined fit","lf");
   entry->SetFillColor(4);
   entry->SetFillStyle(3003);
   entry->SetLineColor(4);
   entry->SetLineStyle(1);
   entry->SetLineWidth(3);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   pad1->Modified();
   c->cd();
  
// ------------>Primitives in pad: pad2
   TPad *pad2 = new TPad("pad2", "pad2",0,0,1,0.66);
   pad2->Draw();
   pad2->cd();
   pad2->Range(-30,-0.1666667,20,6.5);
   pad2->SetFillColor(0);
   pad2->SetBorderMode(0);
   pad2->SetBorderSize(2);
   pad2->SetLeftMargin(0.3);
   pad2->SetTopMargin(0);
   pad2->SetFrameBorderMode(0);
   pad2->SetFrameBorderMode(0);
   
   TH1D *dummy2__2 = new TH1D("dummy2__2","dummy2",6,0.5,6.5);
   dummy2__2->SetBinContent(1,-15);
   dummy2__2->SetBinContent(2,-15);
   dummy2__2->SetBinContent(3,-15);
   dummy2__2->SetBinContent(4,-15);
   dummy2__2->SetBinContent(5,-15);
   dummy2__2->SetBinContent(6,-15);
   dummy2__2->SetBinError(1,15);
   dummy2__2->SetBinError(2,15);
   dummy2__2->SetBinError(3,15);
   dummy2__2->SetBinError(4,15);
   dummy2__2->SetBinError(5,15);
   dummy2__2->SetBinError(6,15);
   dummy2__2->SetMinimum(-15);
   dummy2__2->SetMaximum(15);
   dummy2__2->SetEntries(6);
   dummy2__2->SetStats(0);
   dummy2__2->SetLineColor(0);
   dummy2__2->GetXaxis()->SetTitle("ggF category");
   dummy2__2->GetXaxis()->SetBinLabel(1,"450 < p_{T} < 500 GeV");
   dummy2__2->GetXaxis()->SetBinLabel(2,"500 < p_{T} < 550 GeV");
   dummy2__2->GetXaxis()->SetBinLabel(3,"550 < p_{T} < 600 GeV");
   dummy2__2->GetXaxis()->SetBinLabel(4,"600 < p_{T} < 675 GeV");
   dummy2__2->GetXaxis()->SetBinLabel(5,"675 < p_{T} < 800 GeV");
   dummy2__2->GetXaxis()->SetBinLabel(6,"800 < p_{T} < 1200 GeV");
   dummy2__2->GetXaxis()->SetLabelFont(42);
   dummy2__2->GetXaxis()->SetLabelSize(0.05);
   dummy2__2->GetXaxis()->SetTitleSize(0.05);
   dummy2__2->GetXaxis()->SetTitleOffset(3);
   dummy2__2->GetXaxis()->SetTitleFont(42);
   dummy2__2->GetYaxis()->SetTitle("#mu_{ggF}");
   dummy2__2->GetYaxis()->SetLabelFont(42);
   dummy2__2->GetYaxis()->SetLabelSize(0.05);
   dummy2__2->GetYaxis()->SetTitleSize(0.05);
   dummy2__2->GetYaxis()->SetTitleOffset(0.75);
   dummy2__2->GetYaxis()->SetTitleFont(42);
   dummy2__2->GetZaxis()->SetLabelFont(42);
   dummy2__2->GetZaxis()->SetLabelSize(0.035);
   dummy2__2->GetZaxis()->SetTitleSize(0.035);
   dummy2__2->GetZaxis()->SetTitleFont(42);
   dummy2__2->Draw("hbar");
   
   Double_t Graph0_fx2[2] = {
   1,
   1};
   Double_t Graph0_fy2[2] = {
   -100,
   100};
   graph = new TGraph(2,Graph0_fx2,Graph0_fy2);
   graph->SetName("Graph0");
   graph->SetTitle("Graph");
   graph->SetFillStyle(1000);
   graph->SetLineStyle(3);
   graph->SetLineWidth(3);
   
   TH1F *Graph_Graph_Graph12 = new TH1F("Graph_Graph_Graph12","Graph",100,0.9,2.1);
   Graph_Graph_Graph12->SetMinimum(-120);
   Graph_Graph_Graph12->SetMaximum(120);
   Graph_Graph_Graph12->SetDirectory(0);
   Graph_Graph_Graph12->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph_Graph12->SetLineColor(ci);
   Graph_Graph_Graph12->GetXaxis()->SetLabelFont(42);
   Graph_Graph_Graph12->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph12->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph12->GetXaxis()->SetTitleFont(42);
   Graph_Graph_Graph12->GetYaxis()->SetLabelFont(42);
   Graph_Graph_Graph12->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph12->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph12->GetYaxis()->SetTitleOffset(0);
   Graph_Graph_Graph12->GetYaxis()->SetTitleFont(42);
   Graph_Graph_Graph12->GetZaxis()->SetLabelFont(42);
   Graph_Graph_Graph12->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph_Graph12->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph_Graph12->GetZaxis()->SetTitleFont(42);
   graph->SetHistogram(Graph_Graph_Graph12);
   
   graph->Draw("l");
   
   Double_t Graph0_fx3005[2] = {
   0.093,
   3.308};
   Double_t Graph0_fy3005[2] = {
   0,
   0};
   Double_t Graph0_felx3005[2] = {
   0,
   0};
   Double_t Graph0_fely3005[2] = {
   -100,
   100};
   Double_t Graph0_fehx3005[2] = {
   0,
   0};
   Double_t Graph0_fehy3005[2] = {
   -100,
   100};
   grae = new TGraphAsymmErrors(2,Graph0_fx3005,Graph0_fy3005,Graph0_felx3005,Graph0_fehx3005,Graph0_fely3005,Graph0_fehy3005);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   grae->SetLineWidth(3);
   
   TH1F *Graph_Graph3005 = new TH1F("Graph_Graph3005","Graph",100,0,3.6295);
   Graph_Graph3005->SetMinimum(-120);
   Graph_Graph3005->SetMaximum(120);
   Graph_Graph3005->SetDirectory(0);
   Graph_Graph3005->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3005->SetLineColor(ci);
   Graph_Graph3005->GetXaxis()->SetLabelFont(42);
   Graph_Graph3005->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3005->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3005->GetXaxis()->SetTitleFont(42);
   Graph_Graph3005->GetYaxis()->SetLabelFont(42);
   Graph_Graph3005->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3005->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3005->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3005->GetYaxis()->SetTitleFont(42);
   Graph_Graph3005->GetZaxis()->SetLabelFont(42);
   Graph_Graph3005->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3005->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3005->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3005);
   
   grae->Draw("3");
   
   Double_t Graph1_fx3006[2] = {
   1.613,
   1.613};
   Double_t Graph1_fy3006[2] = {
   -100,
   100};
   Double_t Graph1_felx3006[2] = {
   1.52,
   1.52};
   Double_t Graph1_fely3006[2] = {
   0,
   0};
   Double_t Graph1_fehx3006[2] = {
   1.695,
   1.695};
   Double_t Graph1_fehy3006[2] = {
   0,
   0};
   grae = new TGraphAsymmErrors(2,Graph1_fx3006,Graph1_fy3006,Graph1_felx3006,Graph1_fehx3006,Graph1_fely3006,Graph1_fehy3006);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillColor(4);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   grae->SetLineWidth(3);
   
   TH1F *Graph_Graph3006 = new TH1F("Graph_Graph3006","Graph",100,0,3.6295);
   Graph_Graph3006->SetMinimum(-120);
   Graph_Graph3006->SetMaximum(120);
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
   
   grae->Draw("l");
   
   Double_t Graph2_fx3007[6] = {
   -2.592,
   0.456,
   3.814,
   2.582,
   3.795,
   3.014};
   Double_t Graph2_fy3007[6] = {
   1,
   2,
   3,
   4,
   5,
   6};
   Double_t Graph2_felx3007[6] = {
   2.334,
   2.451,
   2.649,
   2.595,
   3.26,
   4.039};
   Double_t Graph2_fely3007[6] = {
   0,
   0,
   0,
   0,
   0,
   0};
   Double_t Graph2_fehx3007[6] = {
   2.357,
   2.488,
   2.71,
   2.673,
   3.391,
   4.298};
   Double_t Graph2_fehy3007[6] = {
   0,
   0,
   0,
   0,
   0,
   0};
   grae = new TGraphAsymmErrors(6,Graph2_fx3007,Graph2_fy3007,Graph2_felx3007,Graph2_fehx3007,Graph2_fely3007,Graph2_fehy3007);
   grae->SetName("Graph2");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineColor(94);
   grae->SetLineWidth(20);
   grae->SetMarkerSize(0);
   
   TH1F *Graph_Graph3007 = new TH1F("Graph_Graph3007","Graph",100,-6.1498,8.5358);
   Graph_Graph3007->SetMinimum(0.5);
   Graph_Graph3007->SetMaximum(6.5);
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
   
   grae->Draw("p");
   
   Double_t Graph3_fx3008[6] = {
   -2.597,
   0.454,
   3.81,
   2.579,
   3.795,
   3.013};
   Double_t Graph3_fy3008[6] = {
   1,
   2,
   3,
   4,
   5,
   6};
   Double_t Graph3_felx3008[6] = {
   2.823,
   2.648,
   2.741,
   2.71,
   3.388,
   4.532};
   Double_t Graph3_fely3008[6] = {
   0,
   0,
   0,
   0,
   0,
   0};
   Double_t Graph3_fehx3008[6] = {
   2.454,
   2.734,
   3.454,
   3.32,
   4.225,
   5.368};
   Double_t Graph3_fehy3008[6] = {
   0,
   0,
   0,
   0,
   0,
   0};
   grae = new TGraphAsymmErrors(6,Graph3_fx3008,Graph3_fy3008,Graph3_felx3008,Graph3_fehx3008,Graph3_fely3008,Graph3_fehy3008);
   grae->SetName("Graph3");
   grae->SetTitle("Graph");
   grae->SetFillColor(94);
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3008 = new TH1F("Graph_Graph3008","Graph",100,-6.8001,9.7611);
   Graph_Graph3008->SetMinimum(0.5);
   Graph_Graph3008->SetMaximum(6.5);
   Graph_Graph3008->SetDirectory(0);
   Graph_Graph3008->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph3008->SetLineColor(ci);
   Graph_Graph3008->GetXaxis()->SetLabelFont(42);
   Graph_Graph3008->GetXaxis()->SetLabelSize(0.035);
   Graph_Graph3008->GetXaxis()->SetTitleSize(0.035);
   Graph_Graph3008->GetXaxis()->SetTitleFont(42);
   Graph_Graph3008->GetYaxis()->SetLabelFont(42);
   Graph_Graph3008->GetYaxis()->SetLabelSize(0.035);
   Graph_Graph3008->GetYaxis()->SetTitleSize(0.035);
   Graph_Graph3008->GetYaxis()->SetTitleOffset(0);
   Graph_Graph3008->GetYaxis()->SetTitleFont(42);
   Graph_Graph3008->GetZaxis()->SetLabelFont(42);
   Graph_Graph3008->GetZaxis()->SetLabelSize(0.035);
   Graph_Graph3008->GetZaxis()->SetTitleSize(0.035);
   Graph_Graph3008->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_Graph3008);
   
   grae->Draw("p");
   pad2->Modified();
   c->cd();
   c->Modified();
   c->cd();
   c->SetSelected(c);
}
