Set up environment by following instructions at 
https://github.com/CoffeaTeam/lpcjobqueue/

Make sure you have a valid grid certificate

Run over all input files for a particular year by doing
```
python submit-dask.py $year 
```
Run in a screen session or keep the terminal window open.

Follow the progress of dask jobs by opening another terminal window to the same cmlspc node:
```
ssh -L 8787:localhost:8787 USERNAME@cmslpcXXX@fnal.gov
```
Then point your browser to http://localhost:8787/

Coffea output files are put into the outfiles directory. Create a pickle of the histogram with name templates by doing
```
python make-pkl.py $year
```

Make root histograms to feed into combine using the python scripts in vh-scripts.

---
## Notes about dask

* Need to install this before submitting jobs:
```
pip install git+https://github.com/CoffeaTeam/lpcjobqueue.git@v0.2.5
```

* Might need to update `correctionlib` library by doing:
```
pip install --upgrade --user correctionlib
```