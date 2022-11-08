import os, sys
import subprocess
import json
import uproot
import awkward as ak

from coffea import processor, util, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from boostedhiggs import VBFSTXSProcessor

year = sys.argv[1]

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

p = VBFSTXSProcessor(year=year,jet_arbitration='ddb',systematics=False)
args = {'savemetrics':True, 'schema':NanoAODSchema}

this_file = "infiles/"+year+"_VBFHToBB.json"

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

outfile = 'outfiles-stxs/'+str(year)+'_dask_VBFHToBB.coffea'
util.save(output, outfile)
print("saved " + outfile)

