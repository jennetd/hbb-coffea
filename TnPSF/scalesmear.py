import os
import sys
import shutil
import warnings

import re
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from uproot3_methods.classes.TH1 import Methods as TH1Methods
    import uproot3

import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import interp1d
from mplhep.error_estimation import poisson_interval

class AffineMorphTemplate(object):
    def __init__(self, hist):
        '''
        hist: a numpy-histogram-like tuple of (sumw, edges)
        '''
        self.sumw, self.edges = hist
        self.centers = self.edges[:-1] + np.diff(self.edges)/2
        self.norm = self.sumw.sum()
        self.mean = (self.sumw*self.centers).sum() / self.norm
        self.cdf = interp1d(x=self.edges,
                            y=np.r_[0, np.cumsum(self.sumw / self.norm)],
                            kind='linear',
                            assume_sorted=True,
                            bounds_error=False,
                            fill_value=(0, 1),
                           )
        
    def get(self, shift=0., scale=1.):
        '''
        Return a shifted and scaled histogram
        i.e. new edges = edges * scale + shift
        '''
        if not np.isclose(scale, 1.):
            shift += self.mean * (1 - scale)
        scaled_edges = (self.edges - shift) / scale
        return np.diff(self.cdf(scaled_edges)) * self.norm, self.edges

    def scale(self, n):
        self.norm = self.norm * n
     

class MorphHistW2(object):
    def __init__(self, hist):
        '''
        hist: uproot/UHI histogram or a tuple (values, edges, variances)
        '''
        try:
            self.sumw = hist.values
            self.edges = hist.edges
            self.variances = hist.variances
        except:
            self.sumw, self.edges, self.variances = hist
        
        from mplhep.error_estimation import poisson_interval
        down, up = np.nan_to_num(np.abs(poisson_interval(self.sumw, self.variances)), 0.)

        self.nominal = AffineMorphTemplate((self.sumw, self.edges))
        self.w2s = AffineMorphTemplate((self.variances, self.edges))
        
    def get(self, shift=0., scale=1.):
        nom, edges = self.nominal.get(shift, scale)
        w2s, edges = self.w2s.get(shift, scale)       
        return nom, edges, w2s


class TH1(TH1Methods,list):
    pass


class TAxis(object):
    def __init__(self, fNbins, fXmin, fXmax):
        self._fNbins = fNbins
        self._fXmin = fXmin
        self._fXmax = fXmax


def export1d(hist, name='x', label='x', histtype=b"TH1F"):
    """Export a 1-dimensional `Hist` object to uproot

    """
    try:
        sumw, edges, sumw2 = hist
    except:
        sumw, edges = hist
        sumw2 = sumw
    sumw = np.r_[0, sumw, 0]
    sumw2 = np.r_[0, sumw, 0]
 
    out = TH1.__new__(TH1)
    out._fXaxis = TAxis(len(edges) - 1, edges[0], edges[-1])
    out._fXaxis._fName = name
    out._fXaxis._fTitle = label
    out._fXaxis._fXbins = edges.astype(">f8")

    centers = (edges[:-1] + edges[1:]) / 2.0
    out._fEntries = out._fTsumw = out._fTsumw2 = sumw[1:-1].sum()
    out._fTsumwx = (sumw[1:-1] * centers).sum()
    out._fTsumwx2 = (sumw[1:-1] * centers**2).sum()

    out._fName = "histogram"
    out._fTitle = label

    out._classname = histtype.encode()
    out.extend(sumw.astype(">f8"))
    out._fSumw2 = sumw2.astype(">f8")

    return out


def mdev(hist):
    w, edges = hist
    N = np.sum(w)
    centers = edges[:-1] + 0.5*np.diff(edges)
    mean = 1/N * np.sum(w * centers)
    stdev2 = 1/N * np.sum(w * (centers-mean)**2)
    return np.array([mean, np.sqrt(stdev2)])

if __name__ == "__main__":

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

    parser.add_argument('-i', '--in', dest='in_file', required=True, help="Source file")
    parser.add_argument('-o', '--out', dest='out_file', default=None, help="Out file")
    parser.add_argument("--scale", default='1', type=float, help="Scale value.")
    parser.add_argument("--smear", default='0.5', type=float, help="Smear value.")
    parser.add_argument('--plot', action='store_true', help="Make control plots")
    parser.add_argument('--type', dest='hist_type', type=str, choices=["TH1F", "TH1D"], default="TH1D", help="TH1 type. Should be consistent with input.")
    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = args.in_file.replace(".root", "_var.root")
    print("Running with the following options:")
    print(args)

    source_file = uproot3.open(args.in_file)
    if os.path.exists(args.out_file):
        os.remove(args.out_file)
    fout = uproot3.create(args.out_file)

    work_dir = os.path.dirname(args.in_file)

    # scale ttbar templates
    keys = [k.decode("utf-8")[:-2] for k in source_file.keys()]
    for template_name in [k for k in keys if 'match' in k]:
        morph_base = MorphHistW2(source_file[template_name])

        scale_up = morph_base.get(shift=args.scale)
        scale_down = morph_base.get(shift=-args.scale)
        smear_up = morph_base.get(scale=1+args.smear)
        smear_down = morph_base.get(scale=1-args.smear)

        if args.plot:
            import matplotlib.pyplot as plt
            import mplhep as hep

            plt.style.use(hep.style.ROOT)
            fig, ax = plt.subplots()
            hep.histplot(morph_base.get()[:2], color='black' , ls=':', label='Nominal')
            hep.histplot(scale_up[:2], color='blue' , ls='--', label='Up')
            hep.histplot(scale_down[:2], color='red' , ls='--', label='Down')
            ax.set_xlabel('jet $m_{SD}$')
            ax.legend()
            fig.savefig('{}/plot_{}_scale.png'.format(work_dir, template_name))

            fig, ax = plt.subplots()
            hep.histplot(morph_base.get()[:2], color='black' , ls=':', label='Nominal')
            hep.histplot(smear_up[:2], color='blue' , ls='--', label='Up')
            hep.histplot(smear_down[:2], color='red' , ls='--', label='Down')
            ax.set_xlabel('jet $m_{SD}$')
            ax.legend()
            fig.savefig('{}/plot_{}_smear.png'.format(work_dir, template_name))

        fout[template_name] = source_file[template_name]
        fout[template_name.replace("nominal", "smearDown")] = export1d(smear_down, histtype=args.hist_type)
        fout[template_name.replace("nominal", "smearUp")] = export1d(smear_up, histtype=args.hist_type)
        fout[template_name.replace("nominal", "scaleDown")] = export1d(scale_down, histtype=args.hist_type)
        fout[template_name.replace("nominal", "scaleUp")] = export1d(scale_up, histtype=args.hist_type)

        template_name

    # Clone remaining templates:
    for template_name in [k for k in keys if 'match' not in k]:
        fout[template_name] = source_file[template_name]

    fout.close()
