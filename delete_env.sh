#!/bin/bash

[ "$#" -lt 1 ] && { echo "Usage: delete_env.sh <name>"; exit 1; }

base=$(pwd)
env=$1

rm -rf "$base/envs/$env"
rm -rf "$base/envs/jupyter/share/jupyter/kernels/$env"

echo "Successfully deleted $env."
