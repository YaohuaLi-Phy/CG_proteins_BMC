#!/bin/bash
#SBATCH --job-name="bmc"
#SBATCH -A b1030
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -t 48:00:00
#SBATCH -p buyin
#SBATCH --gres=gpu:4

module purge all
module load cuda/cuda_8.0.61_mpich
module load python
module load ruby
cd $SLURM_SUBMIT_DIR

export LD_LIBRARY_PATH=/software/anaconda2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/tdn5879/codes/hoomd-blue/install/hoomd:$LD_LIBRARY_PATH
export PATH=$PATH:/home/tdn5879/codes/hoomd-blue/install/hoomd
export RSEED=$RANDOM
export MERMER=4.0
export MER_TEMP=5.5
export TEMP_TEMP=0.8
export N_HEX1=4
export N_HEX2=1
export N_SCAF=2


# run HOOMD with 2 MPI ranks (as requested by #SBATCH -n) each with 1 GPU (2 GPUs in total)

mpirun -n 8 python replica_exchange.py

