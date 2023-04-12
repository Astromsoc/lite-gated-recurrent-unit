"""
    Script for inferrence (API ver).
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""


# >>>>>>>>>>>>>>>>>>>>>>> FILEPATH TO YOUR MODEL CHECKPOINT <<<<<<<<<<<<<<<<<<<<<<<
CKPT_FILEPATH = <MODEL FILEPATH HERE>
# NOTE: must have the original configuration yaml file copied in the same folder
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

import os
import torch
import argparse

from fastapi import FastAPI
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from src.utils import *
from src.model import *
from src.infer import Inferer



# create FastAPI app
app = FastAPI(title="Infer (grapheme, phoneme) pairs...")


# model initialization
@app.on_event("startup")
async def initiate_inferer():
    # obtain device information
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 
              'cpu')
    global inferer
    inferer = Inferer(ckpt=CKPT_FILEPATH, device=device)
    # set model to eval mode
    inferer.model.eval()


# inferrence
@app.get("/infer_gps")
async def infer_gps(word: str):
    with torch.inference_mode():
        pred_gps = inferer.infer(word)
    return {'inferred_gp_pairs': inferer.gps_to_gpstr(pred_gps)}
