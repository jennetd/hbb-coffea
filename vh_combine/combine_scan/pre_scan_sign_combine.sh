# Create a list of ddb and ddc score
DdbList="0.5 0.8"
DdcList="0.2009233002565048 0.4641588833612782"

# Loop over the thresholds
for i in $DdbList; do
    for j in $DdcList; do
        echo "DDB: $i; DDC: $j"

        #Make the workspace directory
        mkdir -p DDB-$i-DDC-$j/2017/

        #Copy all the stuff needed over
        cp templates.pkl DDB-$i-DDC-$j/2017/
        cp make-hists-1mv-charm.py DDB-$i-DDC-$j/
        cp make-hists-1mv-light.py DDB-$i-DDC-$j/

        cd DDB-$i-DDC-$j/

        ln -sf ../*.json .
        
        #Replace the threshold in make-hist
        sed -i "s/ddbthr =.*/ddbthr = $i/g" make-hists-1mv-*.py
        sed -i "s/ddcthr =.*/ddcthr = $j/g" make-hists-1mv-*.py

        #Run make-hist
        python make-hists-1mv-charm.py 2017
        python make-hists-1mv-light.py 2017

        cd ../
    done
done

