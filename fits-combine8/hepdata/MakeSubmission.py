from hepdata_lib import RootFileReader
from hepdata_lib import Submission, Variable, Table, Uncertainty

from array import array
import numpy as np
import pandas as pd
import json

def read_from_file(poi, filestring):

    search_string = "   r"+poi+" :    "
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

def get_ggF_STXS():

    print("Making ggF STXS table")
    
    with open('../hbb-stxs/allyears/xs.json') as f:
        smxs = json.load(f)

    table = Table("ggF STXS")
    table.description = "ggF STXS"
    table.location = "Table 4 upper, Figure 10 left"

    STXSbin = Variable("ggF STXS bin", is_independent=True, is_binned=False, units="")
    STXSbin.values = ["ggF300to450","ggF450to650","ggF650plus"]
    table.add_variable(STXSbin)

    # MC
    mc = Variable("ggF MC", is_independent=False, is_binned=False, units="")
    mc.values = [smxs['ggF'][str(i)]['nom'] for i in range(1,4)]
    mc_unc = Uncertainty("MC uncertainty", is_symmetric=False)
    mc_unc.set_values_from_intervals(zip([smxs['ggF'][str(i)]['up'] for i in range(1,4)],
                                         [smxs['ggF'][str(i)]['down'] for i in range(1,4)]), 
                                     nominal=mc.values)
    mc.add_uncertainty(mc_unc)
    table.add_variable(mc)

    # Data
    data = Variable("Data", is_independent=False, is_binned=False, units="")

    # Total uncertainty
    logfile = "../hbb-stxs/allyears/logs/fit_batch.out"
    bin1 = read_from_file("ggF300to450",logfile)*smxs['ggF'][str(1)]['nom']
    bin2 = read_from_file("ggF450to650",logfile)*smxs['ggF'][str(2)]['nom']
    bin3 = read_from_file("ggF650plus",logfile)*smxs['ggF'][str(3)]['nom']

    center = np.array([bin1[0],bin2[0],bin3[0]])
    up_unc = np.array([bin1[2],bin2[2],bin3[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1]])
    
    data.values = center
    data_unc = Uncertainty("Data tot uncertainty", is_symmetric=False)
    data_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    # Stat only uncertainty
    logfile = "../hbb-stxs/allyears/logs/fit_batch_stat.out"
    bin1 = read_from_file("ggF300to450",logfile)*smxs['ggF'][str(1)]['nom']
    bin2 = read_from_file("ggF450to650",logfile)*smxs['ggF'][str(2)]['nom']
    bin3 = read_from_file("ggF650plus",logfile)*smxs['ggF'][str(3)]['nom']

    up_unc = np.array([bin1[2],bin2[2],bin3[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1]])
    
    data_stat_unc = Uncertainty("Data stat uncertainty", is_symmetric=False)
    data_stat_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    data.add_uncertainty(data_stat_unc)
    table.add_variable(data)
    
#    table.add_image("HIG-21-020/Figure_010.pdf")

    return table

def get_VBF_STXS():

    print("Making VBF STXS table")
    
    with open('../hbb-stxs/allyears/xs.json') as f:
        smxs = json.load(f)

    table = Table("VBF STXS")
    table.description = "VBF STXS"
    table.location = "Table 4 lower, Figure 10 right"

    STXSbin = Variable("VBF STXS bin", is_independent=True, is_binned=False, units="")
    STXSbin.values = ["VBF1000to1500","VBF1500plus"]
    table.add_variable(STXSbin)

    # MC
    mc = Variable("VBF MC", is_independent=False, is_binned=False, units="")
    mc.values = [smxs['VBF'][str(i)]['nom'] for i in range(1,3)]
    mc_unc = Uncertainty("MC uncertainty", is_symmetric=False)
    mc_unc.set_values_from_intervals(zip([smxs['VBF'][str(i)]['up'] for i in range(1,3)],
                                         [smxs['VBF'][str(i)]['down'] for i in range(1,3)]), 
                                     nominal=mc.values)
    mc.add_uncertainty(mc_unc)
    table.add_variable(mc)

    # Data
    data = Variable("Data", is_independent=False, is_binned=False, units="")

    # Total uncertainty
    logfile = "../hbb-stxs/allyears/logs/fit_batch.out"
    bin8 = read_from_file("VBF1000to1500",logfile)*smxs['VBF'][str(1)]['nom']
    bin9 = read_from_file("VBF1500plus",logfile)*smxs['VBF'][str(2)]['nom']

    center = np.array([bin8[0],bin9[0]])
    up_unc = np.array([bin8[2],bin9[2]])
    do_unc = np.array([bin8[1],bin9[1]])
    
    data.values = center
    data_unc = Uncertainty("Data tot uncertainty", is_symmetric=False)
    data_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    # Stat only uncertainty
    logfile = "../hbb-stxs/allyears/logs/fit_batch_stat.out"
    bin8 = read_from_file("VBF1000to1500",logfile)*smxs['VBF'][str(1)]['nom']
    bin9 = read_from_file("VBF1500plus",logfile)*smxs['VBF'][str(2)]['nom']

    up_unc = np.array([bin8[2],bin9[2]])
    do_unc = np.array([bin8[1],bin9[1]])
        
    data_stat_unc = Uncertainty("Data stat uncertainty", is_symmetric=False)
    data_stat_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    data.add_uncertainty(data_stat_unc)
    table.add_variable(data)

#    table.add_image("HIG-21-020/Figure_010.pdf")

    return table

def get_signal_purity():

    print("Making signal purity table")

    df_2016APV = pd.read_csv('../../notebooks/2016APV/cutflow.csv',index_col=0)
    df_2016 = pd.read_csv('../../notebooks/2016/cutflow.csv',index_col=0)
    df_2017 = pd.read_csv('../../notebooks/2017/cutflow.csv',index_col=0)
    df_2018 = pd.read_csv('../../notebooks/2018/cutflow.csv',index_col=0)

    cutflow = df_2016APV + df_2016 + df_2017 + df_2018
    cutflow.loc['VH'] = cutflow.loc['ZH'] + cutflow.loc['WH']

    procs = ['ggF','VBF','VH','ttH']
    cutflow = cutflow.loc[procs]

    ggfcat = cutflow['ggfpass'].values
    vbfcat = cutflow['vbfpass'].values

    table = Table("Figure 3")
    table.description = "Simulated contribution of each Higgs production process to Run 2 signal"
    table.location = "Figure 3"
#    table.add_image("HIG-21-020/Figure_003.pdf")

    cat = Variable("Category", is_independent=True, is_binned=False, units="GeV")
    cat.values = ["ggFcat","VBFcat"]
    table.add_variable(cat)

    ggf = Variable("ggF", is_independent=False, is_binned=False, units="")
    ggf.values = [ggfcat[0],vbfcat[0]]
    table.add_variable(ggf)

    vbf = Variable("VBF", is_independent=False, is_binned=False, units="")
    vbf.values = [ggfcat[1],vbfcat[1]]
    table.add_variable(vbf)

    vh = Variable("VH", is_independent=False, is_binned=False, units="")
    vh.values = [ggfcat[2],vbfcat[2]]
    table.add_variable(vh)

    tth = Variable("ttH", is_independent=False, is_binned=False, units="")
    tth.values = [ggfcat[3],vbfcat[3]]
    table.add_variable(tth)

    return table
    
def get_JMSJMR_table():

    print("Making JMS/JMR table")

    table = Table("Table 1")
    table.description = "Jet substructure SF, JMS SF, JMR"
    table.location = "Table 1"

    year = Variable("Year", is_independent=True, is_binned=False, units="GeV")
    year.values = ["2016.0","2016.5","2017","2018"]
    table.add_variable(year)

    f_sub = Variable("Substructure SF", is_independent=False, is_binned=False, units="")
    f_sub.values = [0.99, 0.82, 1.05, 0.94]
    f_sub_unc = Uncertainty("Substructure SF uncertainty", is_symmetric=True)
    f_sub_unc.values = [0.16,0.15,0.10,0.08]
    f_sub.add_uncertainty(f_sub_unc)
    table.add_variable(f_sub)

    f_sigma = Variable("JMR SF", is_independent=False, is_binned=False, units="")
    f_sigma.values = [1.13,1.21,1.09,1.07]
    f_sigma_unc = Uncertainty("JMR SF uncertainty", is_symmetric=True)
    f_sigma_unc.values = [0.04,0.03,0.02,0.03]
    f_sigma.add_uncertainty(f_sigma_unc)
    table.add_variable(f_sigma)

    delta_m = Variable("JMS", is_independent=False, is_binned=False, units="MeV")
    delta_m.values = [-240,440,470,-990]
    delta_m_unc = Uncertainty("JMS uncertainty", is_symmetric=True)
    delta_m_unc.values = [410,340,180,250]
    delta_m.add_uncertainty(delta_m_unc)
    table.add_variable(delta_m)

    return table

def get_muoncr_table():

    print("Making table of event yields in muon CR")
    
    table = Table("Table 2")
    table.description = "Muon control region"
    table.location = "Table 2"

    year = Variable("Year", is_independent=True, is_binned=False, units="GeV")
    year.values = ["2016.0","2016.5","2017","2018"]
    table.add_variable(year)

    postfitfail = Variable("DDB fail postfit", is_independent=False, is_binned=False, units="")
    postfitfail.values = [558,527,1411,1748]
    postfitfail_unc = Uncertainty("DDB fail postfit uncertainty", is_symmetric=True)
    postfitfail_unc.values = [29,41,93,100]
    postfitfail.add_uncertainty(postfitfail_unc)
    table.add_variable(postfitfail)

    datafail = Variable("DDB fail data", is_independent=False, is_binned=False, units="")
    datafail.values = [558,527,1411,1748]
    table.add_variable(datafail)

    postfitpass = Variable("DDB pass postfit", is_independent=False, is_binned=False, units="")
    postfitpass.values = [15.0,6.9,27.2,35.2]
    postfitpass_unc = Uncertainty("DDB pass postfit uncertainty", is_symmetric=True)
    postfitpass_unc.values = [3.9,3.2,5.8,7.3]
    postfitpass.add_uncertainty(postfitpass_unc)
    table.add_variable(postfitpass)

    datapass = Variable("DDB pass data", is_independent=False, is_binned=False, units="")
    datapass.values = [15,6,27,34]
    table.add_variable(datapass)

    return table

def get_correlation_table():

    print("Making correlation table")

    # Create a reader for the input file
    reader = RootFileReader("../hbb-stxs/allyears/robustHesseTest.root")

    # Read the histogram, "correlation" is the histogram name
    data = reader.read_hist_2d("h_correlation")

    # Create variable objects
    x = Variable("First Bin", is_independent=True, is_binned=False)
    x.values = ["ggF300to450","ggF450to650","ggF650plus","VBF1000to1500","VBF1500plus"]

    y = Variable("Second Bin", is_independent=True, is_binned=False)
    y.values = ["ggF300to450","ggF450to650","ggF650plus","VBF1000to1500","VBF1500plus"]

    correlation = Variable("Correlation coefficient", is_independent=False, is_binned=False)
    correlation.values = data["z"]

    # Create the table object and add the variables
    table = Table("correlation_coefficients") #Correlation coefficients between STXS bins")
    for var in [x,y,correlation]:
        table.add_variable(var)

    table.description = "Correlation coefficients between the truth bins of the differential cross section measurement."
    table.location = "Auxiliary material"
#    table.add_image("HIG-21-020/figures/correlation.pdf")

    return table

def get_folding_matrix():

    print("Making folding matrix")

    # Create variable objects
    x = Variable("Analysis bin", is_independent=True, is_binned=False)
    x.values = ["ggF_pT_bin_"+str(i) for i in range(1,7)] + ["VBF_mjj_bin_"+str(i) for i in range(1,3)]

    y = Variable("STXS bin", is_independent=True, is_binned=False)
    y.values = ["ggF300to450","ggF450to650","ggF650plus","VBF1000to1500","VBF1500plus"]

    correlation = Variable("Correlation coefficient", is_independent=False, is_binned=False)
    correlation.values = [0.0,0.1,0.5,2.6,4.4,2.5,0.2,0.1,
                          2.4,2.2,1.4,0.8,0.1,0.0,0.3,0.1,
                          0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                          0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,
                          0.1,0.0,0.0,0.0,0.0,0.0,0.2,0.0]

    # Create the table object and add the variables
    table = Table("Folding matrix")
    for var in [x,y,correlation]:
        table.add_variable(var)

    table.description = "Folding matrix showing the acceptance of each analysis bin per STXS bin"
    table.location = "Auxiliary material"
#    table.add_image("HIG-21-020/figures/both_folding.pdf")

    return table

def get_data_mufit():

    print("Making signal strength data table")

    reader = RootFileReader("../hbb-unblind-ewkz/allyears/fitDiagnosticsTest.root")

    table = Table("Signal strength fit")
    table.description = "All signal region bins of the signal strength fit"
    table.location = "Summarized in figs 4-5, 7-8"

    # Processes to include
    procs = ['EWKW','EWKZ','EWKZbb','VBF','VV','WH','Wjets','ZH','Zjets','Zjetsbb','ggF','qcd','singlet','ttH','ttbar']

    # DDB regions
    regions = ["pass","fail"]

    # Categories and differential bins per region
    # name in fitDiagnostics : name in data file
    categories = {}

    categories["fail"] = {"ptbin0ggffail":"ggf_fail_pt1_data_nominal",
                          "ptbin1ggffail":"ggf_fail_pt2_data_nominal",
                          "ptbin2ggffail":"ggf_fail_pt3_data_nominal",
                          "ptbin3ggffail":"ggf_fail_pt4_data_nominal",
                          "ptbin4ggffail":"ggf_fail_pt5_data_nominal",
                          "ptbin5ggffail":"ggf_fail_pt6_data_nominal",
                          "ptbin0vbflofail":"vbf_fail_mjj1_data_nominal",
                          "ptbin0vbfhifail":"vbf_fail_mjj2_data_nominal",
                          #"muonCRfail":"fail_muondata_nominal"
                         }

    categories["pass"] = {"ptbin0ggfpass":"ggf_pass_pt1_data_nominal",
                          "ptbin1ggfpass":"ggf_pass_pt2_data_nominal",
                          "ptbin2ggfpass":"ggf_pass_pt3_data_nominal",
                          "ptbin3ggfpass":"ggf_pass_pt4_data_nominal",
                          "ptbin4ggfpass":"ggf_pass_pt5_data_nominal",
                          "ptbin5ggfpass":"ggf_pass_pt6_data_nominal",
                          "ptbin0vbflopass":"vbf_pass_mjj1_data_nominal",
                          "ptbin0vbfhipass":"vbf_pass_mjj2_data_nominal",
                          #"muonCRpass":"pass_muondata_nominal"
                         }

    # Big loop
    for year in ["2016APV","2016","2017","2018"]:
        print(year)

        reader_data = RootFileReader("../hbb-unblind-ewkz/"+year+"/signalregion.root")
        reader_muondata = RootFileReader("../hbb-unblind-ewkz/"+year+"/muonCR.root")
    
        for region in regions:
            for cat, data_histname in categories[region].items():

                hist1d = {}
                nevents = {}
                nevents_unc = {}
        
                # Fill the 1D histograms
                hist1d['data'] = reader_data.read_hist_1d(data_histname)
                hist1d['TotalBackground'] = reader.read_hist_1d("shapes_fit_s/"+cat+year+"/total_background")

                # y-axis: N events
                # in data
                nevents['data'] = Variable("Number of data events", is_independent=False, is_binned=False, units="")
                nevents['data'].values = hist1d['data']["y"]
                nevents_unc['data'] = Uncertainty("data uncertainty", is_symmetric=True)
                nevents_unc['data'].values = hist1d['data']["dy"]
                nevents['data'].add_uncertainty(nevents_unc['data'])

                # in total background from fit
                nevents['TotalBackground'] = Variable("Number of background events", is_independent=False, is_binned=False, units="")
                # The 7 here is to multiply by the bin width of 7 GeV
                nevents['TotalBackground'].values = [7*n for n in hist1d['TotalBackground']["y"]]
                nevents_unc['TotalBackground'] = Uncertainty("total bkg uncertainty", is_symmetric=True)
                # The 7 here is to multiply by the bin width of 7 GeV
                nevents_unc['TotalBackground'].values = [7*n for n in hist1d['TotalBackground']["dy"]]
                nevents['TotalBackground'].add_uncertainty(nevents_unc['TotalBackground'])

                # Placeholder for DATA
    
                for p in procs:
                    try:
                        hist1d[p] = reader.read_hist_1d("shapes_fit_s/"+cat+year+"/"+p)
                        nevents[p] = Variable("Number of " + p + " events in DDB "+region+" in "+year, is_independent=False, is_binned=False, units="")
                        # The 7 here is to multiply by the bin width of 7 GeV
                        nevents[p].values = [7*n for n in hist1d[p]["y"]]
                        table.add_variable(nevents[p])
                    except:
                        print("Process "+p+" missing from category "+cat+year+"/"+p)

    # x-axis: soft drop mass
    # Only need one copy
    msd = Variable("$m_{SD}$", is_independent=True, is_binned=False, units="GeV")
    msd.values = hist1d['TotalBackground']["x"]
    table.add_variable(msd)

    return table

def get_likelihood_scan():

    print("Making likelihood table")
    
    # Create a reader for the input file
    reader = RootFileReader("../hbb-unblind-ewkz/allyears/NLL.root")

    # Read the histogram, "correlation" is the histogram name
    data = reader.read_hist_2d("h_likelihood")

    # Create variable objects
    x = Variable("mu_ggF", is_independent=True, is_binned=False)
    x.values = data["x"]

    y = Variable("mu_VBF", is_independent=True, is_binned=False)
    y.values = data["y"]

    correlation = Variable("2DeltaNLL", is_independent=False, is_binned=False)
    correlation.values = data["z"]

    # Create the table object and add the variables
    table = Table("Two dimensional 2 Negative log likelihood")
    for var in [x,y,correlation]:
        table.add_variable(var)

    table.description = "Two-dimensional likelihood scan of the VBF and ggF signal strengths. The magnitude represents twice the negative log likelihood difference with respect to the best fit point."
    table.location = "Figure 6"
#    table.add_image("HIG-21-020/Figure_006.pdf")

    return table

def get_signal_strengths():

    table = Table("Table 3")
    table.description = "Measured signal strengths"
    table.location = "Table 3"

    year = Variable("Year", is_independent=True, is_binned=False, units="GeV")
    year.values = ["2016.0","2016.5","2017","2018","Combined"]
    table.add_variable(year)

    mu_ggF = Variable("mu_ggF", is_independent=False, is_binned=False, units="")
    mu_ggF.values = [2.5, 0.6, 3.3, 0.4, 1.6]
    mu_ggF_unc = Uncertainty("mu_ggF uncertainty", is_symmetric=False)
    mu_ggF_unc.set_values_from_intervals(zip([4.7,4.4,3.1,2.6,1.7],[4.3,4.8,2.7,2.7,1.5]),
                                         nominal=mu_ggF.values)
    mu_ggF.add_uncertainty(mu_ggF_unc)
    table.add_variable(mu_ggF)

    mu_VBF = Variable("mu_VBF", is_independent=False, is_binned=False, units="")
    mu_VBF.values = [5.2,5.6,0.8,8.3,4.9]
    mu_VBF_unc = Uncertainty("mu_VBF uncertainty", is_symmetric=False)
    mu_VBF_unc.set_values_from_intervals(zip([4.6,5.8,2.8,3.9,1.9],[3.8,4.2,2.5,3.0,1.6]),
                                         nominal=mu_VBF.values)
    mu_VBF.add_uncertainty(mu_VBF_unc)
    table.add_variable(mu_VBF)

    return table

def get_mu_per_ggFbin():
    table = Table("ggF per-bin fit")
    table.description = "ggF per-bin fit"
    table.location = "Figure 9 lower"

    ggFbin = Variable("ggF_pT_bin", is_independent=True, is_binned=False, units="")
    ggFbin.values = ["ggF_pT_bin_"+str(i) for i in range(1,7)]
    table.add_variable(ggFbin)

    # Data
    data = Variable("Data", is_independent=False, is_binned=False, units="")

    # Total uncertainty
    ggf_logfile = "../hbb-unblind-ewkz/allyears-bybin/logs/fit_batch.out"
    bin1 = read_from_file("ggF1",ggf_logfile)
    bin2 = read_from_file("ggF2",ggf_logfile)
    bin3 = read_from_file("ggF3",ggf_logfile)
    bin4 = read_from_file("ggF4",ggf_logfile)
    bin5 = read_from_file("ggF5",ggf_logfile)
    bin6 = read_from_file("ggF6",ggf_logfile)

    center = np.array([bin1[0],bin2[0],bin3[0],bin4[0],bin5[0],bin6[0]])
    up_unc = np.array([bin1[2],bin2[2],bin3[2],bin4[2],bin5[2],bin6[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1],bin4[1],bin5[1],bin6[1]])
    
    data.values = center
    data_unc = Uncertainty("Data tot uncertainty", is_symmetric=False)
    data_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    # Stat only uncertainty
    ggf_logfile = "../hbb-unblind-ewkz/allyears-bybin/logs/fit_batch_stat.out"
    bin1 = read_from_file("ggF1",ggf_logfile)
    bin2 = read_from_file("ggF2",ggf_logfile)
    bin3 = read_from_file("ggF3",ggf_logfile)
    bin4 = read_from_file("ggF4",ggf_logfile)
    bin5 = read_from_file("ggF5",ggf_logfile)
    bin6 = read_from_file("ggF6",ggf_logfile)

    up_unc = np.array([bin1[2],bin2[2],bin3[2],bin4[2],bin5[2],bin6[2]])
    do_unc = np.array([bin1[1],bin2[1],bin3[1],bin4[1],bin5[1],bin6[1]])
    
    data_stat_unc = Uncertainty("Data stat uncertainty", is_symmetric=False)
    data_stat_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    data.add_uncertainty(data_stat_unc)

    table.add_variable(data)
#    table.add_image("HIG-21-020/Figure_009.pdf")

    return table

def get_mu_per_VBFbin():

    table = Table("VBF per-bin fit")
    table.description = "VBF per-bin fit"
    table.location = "Figure 9 upper"

    VBFbin = Variable("VBF_mjj_bin", is_independent=True, is_binned=False, units="")
    VBFbin.values = ["VBF_mjj_bin_"+str(i) for i in range(1,3)]
    table.add_variable(VBFbin)

    # Data
    data = Variable("Data", is_independent=False, is_binned=False, units="")

    # Total uncertainty
    vbf_logfile = "../hbb-unblind-ewkz/allyears-bybin/logs/fit_batch.out"
    bin7 = read_from_file("VBF7",vbf_logfile)
    bin8 = read_from_file("VBF8",vbf_logfile)

    center = np.array([bin7[0],bin8[0]])
    up_unc = np.array([bin7[2],bin8[2]])
    do_unc = np.array([bin7[1],bin8[1]])
    
    data.values = center
    data_unc = Uncertainty("Data tot uncertainty", is_symmetric=False)
    data_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    # Stat only uncertainty
    vbf_logfile = "../hbb-unblind-ewkz/allyears-bybin/logs/fit_batch_stat.out"
    bin7 = read_from_file("VBF7",vbf_logfile)
    bin8 = read_from_file("VBF8",vbf_logfile)

    up_unc = np.array([bin7[2],bin8[2]])
    do_unc = np.array([bin7[1],bin8[1]])
    
    data_stat_unc = Uncertainty("Data stat uncertainty", is_symmetric=False)
    data_stat_unc.set_values_from_intervals(zip(up_unc,
                                          do_unc), 
                                     nominal=data.values)

    data.add_uncertainty(data_stat_unc)

    table.add_variable(data)
#    table.add_image("HIG-21-020/Figure_009.pdf")

    return table

def main():
    print("HEPData submission for HIG-21-020")

    submission = Submission()

    # Fig 3: Purities
    #submission.add_table(get_signal_purity())
    
    # Tab 1: Scale/smear
    #submission.add_table(get_JMSJMR_table())

    # Figs 4-5, 7-8: full mu fit
    submission.add_table(get_data_mufit())

    # Tab 2: muon control region
    submission.add_table(get_muoncr_table())

    # Tab 3: signal strengths
    submission.add_table(get_signal_strengths())

    # Fig 6: contours
    submission.add_table(get_likelihood_scan())

    # Fig 9: per-bin fit
    submission.add_table(get_mu_per_ggFbin())
    submission.add_table(get_mu_per_VBFbin())

    # Fig 10, Tab 4: STXS data and predictions
    submission.add_table(get_ggF_STXS())
    submission.add_table(get_VBF_STXS())

    # Aux material 1: folding matrix
#    submission.add_table(get_folding_matrix())
    
    # Aux material 2: STXS correlation coefficients
#    submission.add_table(get_correlation_table())

    submission.create_files("HEPData-HIG-21-020",remove_old=True)

if __name__ == "__main__":
    main()
