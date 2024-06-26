# From here forward you need to use python3

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm

import mplhep as hep
plt.style.use(hep.style.ROOT)

import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({'font.size': 22})
divnorm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=8.5)

pycorr = np.load('pycorr.npy')

fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(pycorr,origin='lower',cmap='bwr',norm=divnorm)

ax.set_xticks(np.linspace(0,5,6))
ax.set_yticks(np.linspace(0,5,6))

ax.set_ylim(-0.5,4.5)
ax.set_xlim(-0.5,4.5)

plt.xticks(rotation=45)
#ax.set_xticklabels([k[0] for k in ggf_norm.keys()])

bins = [r'ggF $300<p_T^H<450$ GeV',r'ggF $450<p_T^H<650$ GeV',r'ggF $p_T^H>650$ GeV',r'VBF $1 < m_{jj}< 1.5$ TeV',r'VBF $m_{jj}>1.5$ TeV','extra']

ax.set_xticklabels(bins);
ax.set_yticklabels(bins);
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")

#ax.set_xlabel('ggF STXS bins')

#axcolor = fig.add_axes([0.7,0.12,0.03,0.75])
#t = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 10, 20]
fig.colorbar(im, format="$%.1f$",label='Correlation coefficient',shrink=0.8)

for (j,i),label in np.ndenumerate(pycorr):
    ax.text(i,j,"{:.3f}".format(label),ha='center',va='center')
    
plt.savefig('correlation.png',bbox_inches='tight')
plt.savefig('correlation.pdf',bbox_inches='tight')

plt.show()
