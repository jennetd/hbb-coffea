import sys

if len(sys.argv) < 2:
    print("Give me an argument")
else:

    import scipy
    import scipy.stats
    import numpy as np

    x = float(sys.argv[1])
    print(np.sqrt(scipy.stats.chi2.ppf(scipy.stats.chi2.cdf(x, df=2), df=1)))
