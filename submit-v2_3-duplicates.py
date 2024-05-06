import os, sys
import subprocess
import json
import uproot
import awkward as ak

from coffea import processor, util, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from boostedhiggs import DuplicatesProcessor

from distributed import Client
from lpcjobqueue import LPCCondorCluster

from dask.distributed import performance_report
from dask_jobqueue import HTCondorCluster, SLURMCluster

from datetime import datetime

env_extra = [
    f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
]

#dask.config.set({"distributed.worker.profile.interval": "2d"})

cluster = LPCCondorCluster(
    job_script_prologue=[
        "export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=1d",
        "export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=2d",
    ],
    shared_temp_directory="/tmp",
    transfer_input_files=["boostedhiggs"],
    ship_env=True,
    memory="4GB",
#    image="coffeateam/coffea-dask:0.7.20-fastjet-3.4.0.1-g4cab023"
)

if not os.path.isdir('outfiles-v2_3-duplicates/'):
    os.mkdir('outfiles-v2_3-duplicates')

cluster.adapt(minimum=1, maximum=250)
with Client(cluster) as client:

    print(datetime.now())
    print("Waiting for at least one worker...")  # noqa
    client.wait_for_workers(1)
    print(datetime.now())

    year = sys.argv[1]

    with performance_report(filename="dask-report.html"):

        infiles = subprocess.getoutput("ls infiles/"+year+"_*.json").split()

        for this_file in infiles:

            index = this_file.split("_")[1].split(".json")[0]
            outfile = 'outfiles-v2_3-duplicates/'+str(year)+'_dask_'+index+'.coffea'
            
            if os.path.isfile(outfile):
                print("File " + outfile + " already exists. Skipping.")
                continue
            else:
                print("Begin running " + outfile)
                print(datetime.now())

            uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

            p = DuplicatesProcessor(year=year,jet_arbitration='ddb')
            args = {'savemetrics':True, 'schema':NanoAODSchema}

            output = processor.run_uproot_job(
                this_file,
                treename="Events",
                processor_instance=p,
                executor=processor.dask_executor,
                executor_args={
                    "client": client,
#                    "skipbadfiles": 1,
                    "schema": processor.NanoAODSchema,
                    "treereduction": 2,
                },
                chunksize=50000,
                #        maxchunks=args.max,
            )

            util.save(output, outfile)
            print("saved " + outfile)
            print(datetime.now())
