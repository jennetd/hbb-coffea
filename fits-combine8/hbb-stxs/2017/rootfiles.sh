rm signalregion.root muonCR.root

dir=/uscms/home/jennetd/nobackup/hbb-prod-modes/vbf-nano-2024/2017
dir2=/uscms/home/jennetd/nobackup/hbb-prod-modes/vbf-nano-2024/notebooks/2017

hadd signalregion.root $dir/2mjj*.root $dir/6pt*.root $dir/stxs*.root
hadd muonCR.root $dir/muonCR*.root

cp $dir2/acc_ggF.json .
cp $dir2/acc_VBF.json .
