#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./log_eff_ph/job.out.%j
#SBATCH -e ./log_eff_ph/job.err.%j
# Initial working directory:
#SBATCH -D /u/yxiong/eff-ph
# Job name
#SBATCH -J test_torch
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=yirong.xiong@brain.mpg.de
#SBATCH --time=12:00:00

module purge 
module load anaconda/3/2021.11 
module load gcc/11 openmpi/4 cuda/11.6 
module load pytorch/gpu-cuda-11.6/2.0.0 
module load horovod-pytorch-2.0.0/gpu-cuda-11.6/0.27.0
source /u/yxiong/eff-ph/.venv/bin/activate

export OMP_NUM_THREADS=3

#srun python /u/yxiong/eff-ph/torch_test.py >> log_eff_ph/log_test_gpu
srun python scripts/compute_ph.py >>log_eff_ph/log_toydata
