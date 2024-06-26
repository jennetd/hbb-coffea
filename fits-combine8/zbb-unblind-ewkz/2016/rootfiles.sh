rm signalregion.root muonCR.root

dir=/uscms/home/jennetd/nobackup/hbb-prod-modes/vbf-nano-2024/2016

hadd signalregion.root $dir/2mjj* $dir/6pt*
hadd muonCR.root $dir/muonCR*
