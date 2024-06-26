from __future__ import print_function, division
import sys
import os
import rhalphalib as rl
import numpy as np
import scipy.stats
import pickle
import uproot


def get_templ(f, sample, syst=None, sumw2=True):
    hist_name = sample
    if syst is not None:
        hist_name += "_" + syst
    h_vals = f[hist_name].values()

    h_edges = f[hist_name].axis().edges()
    h_variances = f[hist_name].variances()
    h_key = 'msd'
    if not sumw2:
        return (h_vals, h_edges, h_key)
    else:
        return (h_vals, h_edges, h_key, h_variances)


def test_sfmodel(fittype='single', scale=1, smear=0.1, template_dir={}, fit_dir=None):
    sys_scale = rl.NuisanceParameter('CMS_scale', 'shapeU')
    sys_smear = rl.NuisanceParameter('CMS_smear', 'shapeU')
    lumi = rl.NuisanceParameter('CMS_lumi', 'lnN')
    jecs = rl.NuisanceParameter('CMS_jecs', 'lnN')
    pu = rl.NuisanceParameter('CMS_pu', 'lnN')
    effSF = rl.IndependentParameter('effSF', 1., 0, 3)
    effSF2 = rl.IndependentParameter('effSF_un', 1., 0, 3)
#    effwSF = rl.IndependentParameter('effwSF', 1., 0, 3)
#    effwSF2 = rl.IndependentParameter('effwSF_un', 1., 0, 3)

    msdbins = np.linspace(40, 141.5, 30)
    msd = rl.Observable('msd', msdbins)
    model = rl.Model("sfModel")
    model.t2w_config = ("-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel  --PO verbose "
                        "--PO 'map=.*/effSF:effSF[1, 0, 10]'")

    if fittype == 'single':
        regions = [('single', 'pass'), ('single', 'fail')]
    elif fittype == 'double':
        regions = [('primary', 'fail'), ('secondary', 'pass'), ('secondary', 'fail')]
    else:
        raise NotImplementedError

    for template, region in regions:


        ch = rl.Channel("wsf{}{}".format(template.capitalize(), region.capitalize()))

        wqq_templ = get_templ(template_dir[template], '{}_match'.format(region), 'nominal')
        wqq_sample = rl.TemplateSample("{}_wqq".format(ch.name), rl.Sample.SIGNAL, wqq_templ)
        wqq_sample.setParamEffect(sys_scale, 
                                  get_templ(template_dir[template], '{}_match'.format(region), 'scaleUp', False),
                                  get_templ(template_dir[template], '{}_match'.format(region), 'scaleDown', False),
                                  scale=scale,
                                  )
        wqq_sample.setParamEffect(sys_smear, 
                                  get_templ(template_dir[template], '{}_match'.format(region), 'smearUp', False),
                                  get_templ(template_dir[template], '{}_match'.format(region), 'smearDown', False),
                                  scale=smear,
                                  )
        wqq_sample.setParamEffect(lumi, 1.023)
        wqq_sample.setParamEffect(jecs, 1.02)
        wqq_sample.setParamEffect(pu, 1.05)
        wqq_sample.autoMCStats(lnN=False)
        ch.addSample(wqq_sample)

        qcd_templ = get_templ(template_dir[template], '{}_un'.format(region), 'nominal')
        qcd_sample = rl.TemplateSample("{}_qcd".format(ch.name), rl.Sample.BACKGROUND, qcd_templ)
        qcd_sample.setParamEffect(lumi, 1.023)
        qcd_sample.setParamEffect(jecs, 1.02)
        qcd_sample.setParamEffect(pu, 1.05)
        qcd_sample.autoMCStats(lnN=False)
        ch.addSample(qcd_sample)

        data_obs = get_templ(template_dir[template], '{}_muondata'.format(region), 'nominal')[:-1]
        ch.setObservation(data_obs)

        model.addChannel(ch)

    if fittype == 'single':
        for sample, SF in zip(['wqq', 'qcd'], [effSF, effSF2]):
            pass_sample1 = model['wsfSinglePass'][sample]
            fail_sample = model['wsfSingleFail'][sample]
            pass_fail = pass_sample1.getExpectation(nominal=True).sum() / fail_sample.getExpectation(nominal=True).sum()

            pass_sample1.setParamEffect(SF, 1.0 * SF)
            fail_sample.setParamEffect(SF, (1 - SF) * pass_fail + 1)

    model.renderCombine(fit_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--fit", type=str, choices={'single', 'double'}, default='double',
                        help="Fit type")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Fit directory")
    parser.add_argument("--scale", type=float, default='1',
                        help="Datacard magnitude for scale shift.")
    parser.add_argument("--smear", type=float, default='0.1',
                        help="Datacard magnitude for smear shift.")

    parser.add_argument("-t", "--t1", "--template", type=str, dest='template1',
                        default='newbasew17/wtemplates_n2cvb_var.root',
                        help="Pass(Pass/Pass) templates")

    parser.add_argument("--t2", "--template2", type=str,  dest='template2',
                        default='newbasew17/wtemplates_cvl_var.root',
                        help="Pass(Pass/Pass) templates")

    args = parser.parse_args()
    print("Running with options:")
    print("    ", args)

    if args.out is None:
        fit_dir = 'fit_{}'.format(args.fit)
    else:
        fit_dir = args.out
    if not os.path.exists(fit_dir):
        os.mkdir(fit_dir)

    if args.fit == 'single':
        regions = ['single']
    elif args.fit == 'double':
        regions = ['primary', 'secondary']
    else:
        raise NotImplementedError

    templates = {}    
    for region, path in zip(regions, [args.template1, args.template2]):
        try:
            templates[region] = uproot.open(path)
        except:
            raise ValueError("Expected a root file at", path )

    test_sfmodel(
        fittype=args.fit,
        scale=args.scale,
        smear=args.smear,
        template_dir=templates,
        fit_dir=fit_dir,
        )

    # Dump config
    conf_dict = vars(args)
    import json
    json.dump(conf_dict,
              open("{}/config.json".format(fit_dir), 'w'),
              sort_keys=True,
              indent=4,
              separators=(',', ': '))
