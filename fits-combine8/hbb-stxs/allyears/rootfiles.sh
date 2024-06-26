rm signalregion.root muonCR.root

hadd signalregion.root ../2016APV/signalregion.root ../2016/signalregion.root ../2017/signalregion.root ../2018/signalregion.root
hadd muonCR.root ../2016APV/muonCR.root ../2016/muonCR.root ../2017/muonCR.root ../2018/muonCR.root

ln -s ../2016APV/signalregion.root 2016APV-signalregion.root
ln -s ../2016/signalregion.root 2016-signalregion.root
ln -s ../2017/signalregion.root 2017-signalregion.root
ln -s ../2018/signalregion.root 2018-signalregion.root

ln -s ../2016APV/muonCR.root 2016APV-muonCR.root
ln -s ../2016/muonCR.root 2016-muonCR.root
ln -s ../2017/muonCR.root 2017-muonCR.root
ln -s ../2018/muonCR.root 2018-muonCR.root
