#!/bin/bash

[ ! "$#" -eq 1 ] && { echo "Usage: create_env.sh <name>"; exit 1; }

env=$1

cd /scratch/blochml
. load.sh

if [ ! -d "envs/$env" ]; then
	python -m venv "envs/$env"
fi

. "envs/$env/bin/activate"
pip install wheel setuptools
pip install -r "envs/$env.txt"
