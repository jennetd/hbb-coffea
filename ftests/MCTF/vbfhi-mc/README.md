Make sub-directories here corresponding to each data taking period. 

Using 2017 as an example:
``` 
mkdir 2017
cd 2017
python make_each_ptrho.py
```

This will take some time to run. It will produce a workspace corresponding to each polynomial order up to (0,3) in (pT,rho). 

Once this is finished, do
``` 
./submit.sh
```

This runs 500 toys and calculates the goodness of fit for each polynomial order (0,rho) and (0,rho+1). 
When the jobs have finished, do
``` 
./ftest.sh 0 $rho
```

Start from pT=0,rho=0. If the F-test statistic has p-value < 5%, take the higher order polynomial as baseline and repeat. 
Stop when the higher order polynomials all have p-value > 5%. 