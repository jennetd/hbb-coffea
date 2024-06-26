year=""
if [[ "$PWD" == *"2016APV"* ]]; then
    year="2016APV"
elif [[ "$PWD" == *"2016"* ]]; then
    year="2016"
elif [[ "$PWD" == *"2017"* ]]; then
    year="2017"
elif [[ "$PWD" == *"2018"* ]]; then
    year="2018"
fi

tag=`echo $PWD | sed 's:.*/::'`
echo $tag

if [ `ls *GoodnessOfFit*.root | wc -l` -lt 1 ]
then
    cp /eos/uscms/store/user/jennetd/f-tests/hbb-f-tests-pf/vbf-mc/${year}/*/${tag}*/*GoodnessOfFit*.root .
fi

rm *total.root
hadd higgsCombineToys.baseline.GoodnessOfFit.mH125.total.root higgsCombineToys.baseline.GoodnessOfFit.mH125.*.root
hadd higgsCombineToys.alternative.GoodnessOfFit.mH125.total.root higgsCombineToys.alternative.GoodnessOfFit.mH125.*.root

tag1=`echo $tag | sed 's/_vs_.*//'`
tag2=`echo $tag | sed 's/.*_vs_//'`
cp ../$tag1/higgsCombineObserved.GoodnessOfFit.mH125.root baseline_obs.root
cp ../$tag2/higgsCombineObserved.GoodnessOfFit.mH125.root alternative_obs.root
