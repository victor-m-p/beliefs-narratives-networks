#!/bin/bash

# create and activate virtual environment
python -m venv otree-env
source otree-env/bin/activate

# upgrade pip
pip install --upgrade pip
pip install -r requirements.txt