#!/bin/bash
#SBATCH -o output-gpu-%A.%a.out
#SBATCH -p serial
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00

export OPENMM_PLUGIN_DIR=/home/steboss/local/openmm/lib/plugins

srun  python vanishing.py

