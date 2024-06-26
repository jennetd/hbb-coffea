year=$1

scale="3"
smear="0.5"

python scalesmear.py -i templates/$year/TnPtemplates.root --plot --scale $scale --smear $smear

python sf.py --fit single -t templates/${year}/TnPtemplates_var.root -o ${year}-FitSingle --scale $scale --smear $smear
cd ${year}-FitSingle
. build.sh
combine -M FitDiagnostics --expectSignal 1 -d model_combined.root --saveShapes --saveWithUncertainties --rMin 0 --rMax 2
python ../results.py --year=${year} --scale $scale --smear $smear
