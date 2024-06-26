# W TagAndProbe rhalphalib implementation

### Setup combine and rhalphalib
Follow official combine instruction or setup with `conda` using [this branch](https://github.com/andrzejnovak/HiggsAnalysis-CombinedLimit/tree/root6.22-compat).

Setup [rhalphalib](https://github.com/nsmith-/rhalphalib) and then

```bash
git clone https://github.com/jennetd/TnPSF.git
cd TnPSF
```

### Generate variations 
For each root file generate variations (only matched - catp2)
```bash
python scalesmear.py -i templates/2016/TnPtemplates.root --plot --scale 4 --smear 0.5
```

### Generate combine/rhalphalib workspace and fit

```bash
python sf.py --fit single -t templates/2016/TnPtemplates_var.root -o 2016-FitSingle --scale 4 --smear 0.5
cd FitSingle
```

To make plots from `FitDiagnostics` output, run within the fit folder:
```
python ../results.py --year 2018
```

