# Determine the year from the directory name

year=""

if [[ "$PWD" == *"2016"* ]]; then
    year="_2016"
elif [[ "$PWD" == *"2017"* ]]; then
year="_2017"
elif [[ "$PWD" == *"2018"* ]]; then
    year="_2018"
fi

echo "VH SIGNIFICANCE"
combine -M Significance -m 125 --signif output/testModel${year}/model_combined.root --cminDefaultMinimizerStrategy 0 -t -1 --redefineSignalPOI rVH --setParameters rVH=1
