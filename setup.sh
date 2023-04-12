#!/bin/bash

# [1] create conda env
yes | conda create -n lgru python=3.9 && conda activate lgru

# [2] install dependencies
pip install -r requirements.txt
yes | conda install tmux

# [3] log into wandb for future archiving
# NOTE: remember to put your wandb API key in `.env` 
wandb login $(cat .env)