import os, sys
import subprocess
import json
import uproot
import awkward as ak

from coffea import processor, util, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from boostedhiggs import BTagEfficiency

from distributed import Client
from lpcjobqueue import LPCCondorCluster

from dask.distributed import performance_report
from dask_jobqueue import HTCondorCluster, SLURMCluster

env_extra = [
    f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
]

cluster = LPCCondorCluster(
    transfer_input_files=["boostedhiggs"],
    ship_env=True,
    memory="4GB",
#    image="coffeateam/coffea-dask:0.7.11-fastjet-3.3.4.0rc9-ga05a1f8",
)

cluster.adapt(minimum=1, maximum=50)
client = Client(cluster)

print("Waiting for at least one worker...")  # noqa
client.wait_for_workers(1)

year = sys.argv[1]

with performance_report(filename="dask-report.html"):

    # get list of input files                                                                                                 
    infiles = subprocess.getoutput("ls infiles/"+year+"*.json").split()

    thefiles = [f for f in infiles if 'QCD' in f or 'TTbar' in f]

    filedict = {}
    for afile in thefiles:
        with open(afile) as f:
            tmpdict = json.load(f)
        filedict.update(tmpdict)

    uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

    p = BTagEfficiency(year=year)
    args = {'savemetrics':True, 'schema':NanoAODSchema}

    output = processor.run_uproot_job(
        filedict,
        treename="Events",
        processor_instance=p,
        executor=processor.dask_executor,
        executor_args={
            "client": client,
            "skipbadfiles": 1,
            "schema": processor.NanoAODSchema,
            "treereduction": 2,
        },
        chunksize=100000,
        #        maxchunks=args.max,
    )

    outfile = 'outfiles-btag/btag_deepJet_M_'+str(year)+'.coffea'
    util.save(output, outfile)
    print("saved " + outfile)

