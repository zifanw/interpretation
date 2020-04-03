#!/bin/bash

echo Start installing the requirements with conda ... 
yes | conda env create -f explain_env.yml
conda activate explain 
python -m ipykernel install --user --name explain
yes | pip install opencv-python

echo Complete the environment installation 
