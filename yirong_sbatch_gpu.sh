#!/bin/bash
#SBATCH -N 1
#SBATCH -p turing
#SBATCH -J xiongy_testing
#SBATCH -D /gpfs/laur/data/xiongy/eff-ph
#SBATCH -w lnx-cm-21007.mpibr.local
#SBATCH -o ./log_eff_ph/job.out.%j
#SBATCH -e ./log_eff_ph/job.err.%j

source /gpfs/laur/data/xiongy/eff-ph/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=3 #specifies the GPU to use on the node

python3 /gpfs/laur/data/xiongy/eff-ph/scripts/compute_ph.py
