pt=$1
rho=$2

mkdir pt${pt}rho${rho}_vs_pt${pt}rho$((${rho}+1))
cd pt${pt}rho${rho}_vs_pt${pt}rho$((${rho}+1))
ln -s ../../copy.sh .
./copy.sh

ln -s ../../../../plot-ftest.py .
python plot-ftest.py

cd ..

mkdir pt${pt}rho${rho}_vs_pt$((${pt}+1))rho${rho}
cd pt${pt}rho${rho}_vs_pt$((${pt}+1))rho${rho}
ln -s ../../copy.sh .
./copy.sh

ln -s ../../../../plot-ftest.py .
python plot-ftest.py

cd ..
