# Create a list of ddb and ddc score
DdbList="0.5 0.8"
DdcList="0.2009233002565048 0.4641588833612782"

cmsenv

# Loop over the thresholds
for i in $DdbList; do
    for j in $DdcList; do
        echo "DDB: $i; DDC: $j" 

        cd DDB-$i-DDC-$j/2017

        SignalRootFile=signalregion.root
        if [ -f "$SignalRootFile" ]; then
            echo "$SignalRootFile exists."
        else 
            hadd signalregion.root 1mvc-signalregion.root 1mvl-signalregion.root
        fi

        cd ..

        ln -sf ../make_cards.py .
        
        #Run make-cards
        #python make_cards.py 2017

        #Run combine
        cd 2017
        ln -sf ../../make_workspace.sh .
        ln -sf ../../exp_significance.sh .

        pwd
        ls 

        #source make_workspace.sh
        source exp_significance.sh > sign.out 2>&1

        cd ../../

        pwd
    done
done



# Replace the number in the script
