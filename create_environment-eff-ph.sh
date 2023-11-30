module purge 
module load anaconda/3/2021.11 
module load gcc/11 openmpi/4 cuda/11.6 
module load pytorch/gpu-cuda-11.6/2.0.0
module load horovod-pytorch-2.0.0/gpu-cuda-11.6/0.27.0
python -m venv --system-site-packages /u/yxiong/eff-ph/.venv
source /u/yxiong/eff-ph/.venv/bin/activate
pip install -r /u/yxiong/eff-ph/requirement_pip.txt
