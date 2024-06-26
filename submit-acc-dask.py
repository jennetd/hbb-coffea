import os, sys
import subprocess
import json
import uproot
import awkward as ak

from coffea import processor, util, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from boostedhiggs import AccProcessor

from distributed import Client
from lpcjobqueue import LPCCondorCluster

from dask.distributed import performance_report
from dask_jobqueue import HTCondorCluster, SLURMCluster

from datetime import datetime

def main():

    env_extra = [
        f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
    ]

    cluster = LPCCondorCluster(
#        job_script_prologue=[
#            "export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=1d",
#            "export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=2d",
        #       ],
        shared_temp_directory="/tmp",
        transfer_input_files=["boostedhiggs"],
        memory="8GB",
    )

    if not os.path.isdir('outfiles-acc/'):
        os.mkdir('outfiles-acc')

    cluster.adapt(minimum=1, maximum=200)
    with Client(cluster) as client:

        print(datetime.now())
        print("Waiting for at least one worker...")  # noqa
        client.wait_for_workers(1)
        print(datetime.now())
        
        year = sys.argv[1]
        
        with performance_report(filename="dask-report.html"):
            
            infiles = subprocess.getoutput("ls infiles-nano/"+year+"_*.json").split()
            
            for this_file in infiles:
                
                index = this_file.split("_")[1].split(".json")[0]
                outfile = 'outfiles-acc/'+str(year)+'_dask_'+index+'.coffea'
                
                if "ggF" not in index and "VBF" not in index:
                    continue
                
                if os.path.isfile(outfile):
                    print("File " + outfile + " already exists. Skipping.")
                    continue
                else:
                    print("Begin running " + outfile)
                    print(datetime.now())

                uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

                p = AccProcessor(ewkHcorr=True,systematics=True)
                args = {'savemetrics':True, 'schema':NanoAODSchema}

                output = processor.run_uproot_job(
                    this_file,
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

                util.save(output, outfile)
                print("saved " + outfile)
                print(datetime.now())

if __name__ == "__main__":
    main()
