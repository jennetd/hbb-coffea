void vbfxs()
{
//=========Macro generated from canvas: vbfxs/vbfxs
//=========  (Tue Oct 10 09:13:55 2023) by ROOT version 6.12/07
   TCanvas *vbfxs = new TCanvas("vbfxs", "vbfxs",73,1103,600,600);
   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   vbfxs->SetHighLightColor(2);
   vbfxs->Range(0,0,1,1);
   vbfxs->SetFillColor(0);
   vbfxs->SetBorderMode(0);
   vbfxs->SetBorderSize(2);
   vbfxs->SetFrameBorderMode(0);
  
// ------------>Primitives in pad: pad1
   TPad *pad1 = new TPad("pad1", "pad1",0,0.33,1,1);
   pad1->Draw();
   pad1->cd();
   pad1->Range(300,1.176072,4966.667,3.151817);
   pad1->SetFillColor(0);
   pad1->SetBorderMode(0);
   pad1->SetBorderSize(2);
   pad1->SetLogy();
   pad1->SetLeftMargin(0.15);
   pad1->SetBottomMargin(1e-05);
   pad1->SetFrameBorderMode(0);
   pad1->SetFrameBorderMode(0);
   Double_t xAxis1[3] = {1000, 1500, 4500}; 
   
   TH1D *dummy1__1 = new TH1D("dummy1__1","dummy1",2, xAxis1);
   dummy1__1->SetMinimum(15);
   dummy1__1->SetMaximum(900);
   dummy1__1->SetStats(0);
   dummy1__1->SetLineColor(0);
   dummy1__1->SetLineWidth(3);
   dummy1__1->GetXaxis()->SetTitle("m_{jj}^{gen} [GeV]");
   dummy1__1->GetXaxis()->SetLabelFont(42);
   dummy1__1->GetXaxis()->SetLabelSize(0.04160383);
   dummy1__1->GetXaxis()->SetTitleSize(0.04160383);
   dummy1__1->GetXaxis()->SetTitleOffset(1.34);
   dummy1__1->GetXaxis()->SetTitleFont(42);
   dummy1__1->GetYaxis()->SetTitle("#sigma (fb)");
   dummy1__1->GetYaxis()->SetLabelFont(42);
   dummy1__1->GetYaxis()->SetLabelSize(0.04160383);
   dummy1__1->GetYaxis()->SetTitleSize(0.04160383);
   dummy1__1->GetYaxis()->SetTitleOffset(1.34);
   dummy1__1->GetYaxis()->SetTitleFont(42);
   dummy1__1->GetZaxis()->SetLabelFont(42);
   dummy1__1->GetZaxis()->SetLabelSize(0.035);
   dummy1__1->GetZaxis()->SetTitleSize(0.035);
   dummy1__1->GetZaxis()->SetTitleFont(42);
   dummy1__1->Draw("");
   
   Double_t Graph0_fx3001[2] = {
   1250,
   3000};
   Double_t Graph0_fy3001[2] = {
   30.67989,
   40.70797};
   Double_t Graph0_felx3001[2] = {
   250,
   1500};
   Double_t Graph0_fely3001[2] = {
   1.659813,
   3.133862};
   Double_t Graph0_fehx3001[2] = {
   250,
   1500};
   Double_t Graph0_fehy3001[2] = {
   1.354793,
   2.424485};
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(2,Graph0_fx3001,Graph0_fy3001,Graph0_felx3001,Graph0_fehx3001,Graph0_fely3001,Graph0_fehy3001);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = 1179;
   color = new TColor(ci, 0, 0, 1, " ", 0.1);
   grae->SetFillColor(ci);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   grae->SetLineWidth(3);
   
   TH1F *Graph_Graph3001 = new TH1F("Graph_Graph3001","Graph",100,650,4850);
   Graph_Graph3001->SetMinimum(27.60884);
   Graph_Graph3001->SetMaximum(44.54369);
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
   
   grae->Draw("2");
   
   Double_t Graph0_fx3002[2] = {
   1250,
   3000};
   Double_t Graph0_fy3002[2] = {
   30.67989,
   40.70797};
   Double_t Graph0_felx3002[2] = {
   250,
   1500};
   Double_t Graph0_fely3002[2] = {
   1.659813,
   3.133862};
   Double_t Graph0_fehx3002[2] = {
   250,
   1500};
   Double_t Graph0_fehy3002[2] = {
   1.354793,
   2.424485};
   grae = new TGraphAsymmErrors(2,Graph0_fx3002,Graph0_fy3002,Graph0_felx3002,Graph0_fehx3002,Graph0_fely3002,Graph0_fehy3002);
   grae->SetName("Graph0");
   grae->SetTitle("Graph");

   ci = 1179;
   color = new TColor(ci, 0, 0, 1, " ", 0.1);
   grae->SetFillColor(ci);
   grae->SetFillStyle(3003);
   grae->SetLineColor(4);
   grae->SetLineWidth(3);
   
   TH1F *Graph_Graph_Graph30013002 = new TH1F("Graph_Graph_Graph30013002","Graph",100,650,4850);
   Graph_Graph_Graph30013002->SetMinimum(27.60884);
   Graph_Graph_Graph30013002->SetMaximum(44.54369);
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
   
   Double_t Graph1_fx3003[2] = {
   1250,
   3000};
   Double_t Graph1_fy3003[2] = {
   261.0859,
   116.0584};
   Double_t Graph1_felx3003[2] = {
   250,
   1500};
   Double_t Graph1_fely3003[2] = {
   193.1299,
   59.67788};
   Double_t Graph1_fehx3003[2] = {
   250,
   1500};
   Double_t Graph1_fehy3003[2] = {
   208.2244,
   66.06903};
   grae = new TGraphAsymmErrors(2,Graph1_fx3003,Graph1_fy3003,Graph1_felx3003,Graph1_fehx3003,Graph1_fely3003,Graph1_fehy3003);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3003 = new TH1F("Graph_Graph3003","Graph",100,650,4850);
   Graph_Graph3003->SetMinimum(15.08755);
   Graph_Graph3003->SetMaximum(510.6033);
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
   
   Double_t Graph2_fx1[2] = {
   -100,
   100};
   Double_t Graph2_fy1[2] = {
   0,
   0};
   TGraph *graph = new TGraph(2,Graph2_fx1,Graph2_fy1);
   graph->SetName("Graph2");
   graph->SetTitle("Graph");
   graph->SetFillStyle(1000);
   graph->SetLineColor(0);
   graph->SetLineWidth(3);
   
   TH1F *Graph_Graph1 = new TH1F("Graph_Graph1","Graph",100,-120,120);
   Graph_Graph1->SetMinimum(0.0011);
   Graph_Graph1->SetMaximum(1.1);
   Graph_Graph1->SetDirectory(0);
   Graph_Graph1->SetStats(0);

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
   TLatex *   tex = new TLatex(0.67,0.92,"138 fb^{-1} (13 TeV)");
tex->SetNDC();
   tex->SetTextFont(42);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.19,0.82,"CMS");
tex->SetNDC();
   tex->SetLineWidth(2);
   tex->Draw();
   
   TLegend *leg = new TLegend(0.6,0.7,0.85,0.87,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.04160383);
   leg->SetLineColor(0);
   leg->SetLineStyle(0);
   leg->SetLineWidth(0);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("Graph0","SM (Powheg)","f");

   ci = 1179;
   color = new TColor(ci, 0, 0, 1, " ", 0.1);
   entry->SetFillColor(ci);
   entry->SetFillStyle(3003);
   entry->SetLineColor(4);
   entry->SetLineStyle(1);
   entry->SetLineWidth(3);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(21);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   entry=leg->AddEntry("Graph1","Data","p");
   entry->SetLineColor(1);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);
   entry->SetMarkerColor(1);
   entry->SetMarkerStyle(20);
   entry->SetMarkerSize(1);
   entry->SetTextFont(42);
   leg->Draw();
   pad1->Modified();
   vbfxs->cd();
  
// ------------>Primitives in pad: pad2
   TPad *pad2 = new TPad("pad2", "pad2",0,0,1,0.33);
   pad2->Draw();
   pad2->cd();
   pad2->Range(300,-6.857241,4966.667,16.00023);
   pad2->SetFillColor(0);
   pad2->SetBorderMode(0);
   pad2->SetBorderSize(2);
   pad2->SetLeftMargin(0.15);
   pad2->SetTopMargin(1e-05);
   pad2->SetBottomMargin(0.3);
   pad2->SetFrameBorderMode(0);
   pad2->SetFrameBorderMode(0);
   Double_t xAxis2[3] = {1000, 1500, 4500}; 
   
   TH1D *dummy2__2 = new TH1D("dummy2__2","dummy1",2, xAxis2);
   dummy2__2->SetMinimum(0);
   dummy2__2->SetMaximum(16);
   dummy2__2->SetStats(0);
   dummy2__2->SetLineColor(0);
   dummy2__2->SetLineWidth(3);
   dummy2__2->GetXaxis()->SetTitle("m_{jj}^{gen} [GeV]");
   dummy2__2->GetXaxis()->SetLabelFont(42);
   dummy2__2->GetXaxis()->SetLabelSize(0.08446838);
   dummy2__2->GetXaxis()->SetTitleSize(0.08446838);
   dummy2__2->GetXaxis()->SetTitleOffset(1.34);
   dummy2__2->GetXaxis()->SetTitleFont(42);
   dummy2__2->GetYaxis()->SetTitle("Ratio to SM");
   dummy2__2->GetYaxis()->SetLabelFont(42);
   dummy2__2->GetYaxis()->SetLabelSize(0.08446838);
   dummy2__2->GetYaxis()->SetTitleSize(0.08446838);
   dummy2__2->GetYaxis()->SetTitleOffset(0.66);
   dummy2__2->GetYaxis()->SetTitleFont(42);
   dummy2__2->GetZaxis()->SetLabelFont(42);
   dummy2__2->GetZaxis()->SetLabelSize(0.035);
   dummy2__2->GetZaxis()->SetTitleSize(0.035);
   dummy2__2->GetZaxis()->SetTitleFont(42);
   dummy2__2->Draw("");
   
   Double_t Graph0_fx3004[3] = {
   1250,
   3000,
   1.58101e-322};
   Double_t Graph0_fy3004[3] = {
   1,
   1,
   1};
   Double_t Graph0_felx3004[3] = {
   250,
   1500,
   4.680888e-316};
   Double_t Graph0_fely3004[3] = {
   0.204679,
   0.205062,
   0.212264};
   Double_t Graph0_fehx3004[3] = {
   250,
   1500,
   4.680888e-316};
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
   grae->SetLineWidth(0);
   
   TH1F *Graph_Graph3004 = new TH1F("Graph_Graph3004","Graph",100,0,4950);
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
   1250,
   3000,
   1.58101e-322};
   Double_t Graph0_fy3005[3] = {
   1,
   1,
   1};
   Double_t Graph0_felx3005[3] = {
   250,
   1500,
   4.680888e-316};
   Double_t Graph0_fely3005[3] = {
   0.204679,
   0.205062,
   0.212264};
   Double_t Graph0_fehx3005[3] = {
   250,
   1500,
   4.680888e-316};
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
   grae->SetLineWidth(0);
   
   TH1F *Graph_Graph_Graph30043005 = new TH1F("Graph_Graph_Graph30043005","Graph",100,0,4950);
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
   
   Double_t Graph1_fx3006[2] = {
   1250,
   3000};
   Double_t Graph1_fy3006[2] = {
   8.51,
   2.851};
   Double_t Graph1_felx3006[2] = {
   250,
   1500};
   Double_t Graph1_fely3006[2] = {
   6.295,
   1.466};
   Double_t Graph1_fehx3006[2] = {
   250,
   1500};
   Double_t Graph1_fehy3006[2] = {
   6.787,
   1.623};
   grae = new TGraphAsymmErrors(2,Graph1_fx3006,Graph1_fy3006,Graph1_felx3006,Graph1_fehx3006,Graph1_fely3006,Graph1_fehy3006);
   grae->SetName("Graph1");
   grae->SetTitle("Graph");
   grae->SetFillStyle(1000);
   grae->SetLineWidth(3);
   grae->SetMarkerStyle(20);
   
   TH1F *Graph_Graph3006 = new TH1F("Graph_Graph3006","Graph",100,650,4850);
   Graph_Graph3006->SetMinimum(1.2465);
   Graph_Graph3006->SetMaximum(16.6882);
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
   
   grae->Draw("p");
   pad2->Modified();
   vbfxs->cd();
   vbfxs->Modified();
   vbfxs->cd();
   vbfxs->SetSelected(vbfxs);
}
