import os, sys
import subprocess
import json
import uproot
import awkward as ak
import pandas as pd

from coffea import processor, util, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from boostedhiggs import VBFProcessor

year = sys.argv[1]

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

if not os.path.isdir('outfiles/'):
    os.mkdir('outfiles')

p = VBFProcessor(year=year,jet_arbitration='ddb')
args = {'savemetrics':True, 'schema':NanoAODSchema}

this_file = "infiles/"+year+"_EWKV.json"
outfile = 'outfiles/'+str(year)+'_dask_EWKV.coffea'

output = processor.run_uproot_job(
    this_file,
    treename="Events",
    processor_instance=p,
    executor=processor.iterative_executor,
    executor_args={
        "skipbadfiles": 1,
        "schema": processor.NanoAODSchema,
    },
    chunksize=10000,
#    maxchunks=1,                                                                                                                               
)

util.save(output, outfile)
print("saved " + outfile)
