void mu_bybin()
{
//=========Macro generated from canvas: c/c
//=========  (Fri May 31 09:25:49 2024) by ROOT version 6.12/07
   TCanvas *c = new TCanvas("c", "c",0,23,800,800);
   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   c->SetHighLightColor(2);
   c->Range(-0.6000001,-1.1,1.4,9.566667);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetLeftMargin(0.3);
   c->SetBottomMargin(0.15);
   c->SetFrameBorderMode(0);
   c->SetFrameBorderMode(0);
   
   TH1D *dummy1__1 = new TH1D("dummy1__1","dummy1",8,0.5,8.5);
   dummy1__1->SetBinContent(1,-15);
   dummy1__1->SetBinContent(2,-15);
   dummy1__1->SetBinContent(3,-15);
   dummy1__1->SetBinContent(4,-15);
   dummy1__1->SetBinContent(5,-15);
   dummy1__1->SetBinContent(6,-15);
   dummy1__1->SetBinContent(7,-15);
   dummy1__1->SetBinContent(8,-15);
   dummy1__1->SetBinError(1,15);
   dummy1__1->SetBinError(2,15);
   dummy1__1->SetBinError(3,15);
   dummy1__1->SetBinError(4,15);
   dummy1__1->SetBinError(5,15);
   dummy1__1->SetBinError(6,15);
   dummy1__1->SetBinError(7,15);
   dummy1__1->SetBinError(8,15);
   dummy1__1->SetMinimum(0);
   dummy1__1->SetMaximum(1.2);
   dummy1__1->SetEntries(8);
   dummy1__1->SetStats(0);
   dummy1__1->SetLineColor(0);
   dummy1__1->GetXaxis()->SetBinLabel(1,"1000 < m_{jj} < 2000 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(2,"m_{jj} > 2000 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(3,"450 < p_{T} < 500 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(4,"500 < p_{T} < 550 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(5,"550 < p_{T} < 600 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(6,"600 < p_{T} < 675 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(7,"675 < p_{T} < 800 GeV");
   dummy1__1->GetXaxis()->SetBinLabel(8,"800 < p_{T} < 1200 GeV");
   dummy1__1->GetXaxis()->SetLabelFont(42);
   dummy1__1->GetXaxis()->SetTitleFont(42);
   dummy1__1->GetYaxis()->SetTitle("#mu_{Zbb}");
   dummy1__1->GetYaxis()->SetLabelFont(42);
   dummy1__1->GetYaxis()->SetTitleOffset(1.5);
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
   0.511,
   0.759};
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
   
   TH1F *Graph_Graph3001 = new TH1F("Graph_Graph3001","Graph",100,0.4862,0.7838);
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
   0.622,
   0.622};
   Double_t Graph2_fy3002[2] = {
   -100,
   100};
   Double_t Graph2_felx3002[2] = {
   0.111,
   0.111};
   Double_t Graph2_fely3002[2] = {
   0,
   0};
   Double_t Graph2_fehx3002[2] = {
   0.137,
   0.137};
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
   
   TH1F *Graph_Graph3002 = new TH1F("Graph_Graph3002","Graph",100,0.4862,0.7838);
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
   
   Double_t Graph3_fx3003[8] = {
   0.649,
   0.665,
   0.725,
   0.63,
   0.795,
   0.821,
   0.286,
   0.568};
   Double_t Graph3_fy3003[8] = {
   1,
   2,
   3,
   4,
   5,
   6,
   7,
   8};
   Double_t Graph3_felx3003[8] = {
   0.12,
   0.128,
   0.143,
   0.133,
   0.172,
   0.223,
   0.201,
   0.312};
   Double_t Graph3_fely3003[8] = {
   0,
   0,
   7.9055e-315,
   6.907801e-310,
   2.371515e-322,
   6.373447e-322,
   4.16005e-316,
   0};
   Double_t Graph3_fehx3003[8] = {
   0.158,
   0.168,
   0.188,
   0.178,
   0.228,
   0.298,
   0.229,
   0.374};
   Double_t Graph3_fehy3003[8] = {
   0,
   0,
   5.681755e-322,
   7.954457e-322,
   6.907799e-310,
   3.131513e-294,
   6.907799e-310,
   0};
   grae = new TGraphAsymmErrors(8,Graph3_fx3003,Graph3_fy3003,Graph3_felx3003,Graph3_fehx3003,Graph3_fely3003,Graph3_fehy3003);
   grae->SetName("Graph3");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3003 = new TH1F("Graph_Graph3003","Graph",100,0,1.2224);
   Graph_Graph3003->SetMinimum(0.3);
   Graph_Graph3003->SetMaximum(8.7);
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
   TLatex *   tex = new TLatex(0.63,0.92,"138 fb^{-1} (13 TeV)");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.32,0.92,"#bf{CMS}");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.35,0.3,0.5,0.45,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetLineColor(0);
   leg->SetLineStyle(0);
   leg->SetLineWidth(0);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("Graph3","Observed","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph3","#pm1#sigma (stat #oplus syst)","l");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(3);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph2","Combined fit","lf");
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
   c->Modified();
   c->cd();
   c->SetSelected(c);
}
