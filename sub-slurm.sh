#!/bin/bash
#SBATCH --job-name="bmc"
#SBATCH -A b1030
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -p buyin
#SBATCH --gres=gpu:p100:1

module purge all
module load cuda/cuda_8.0.61_mpich
module load python

cd $SLURM_SUBMIT_DIR

export LD_LIBRARY_PATH=/software/anaconda2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/tdn5879/codes/hoomd-blue/install/hoomd:$LD_LIBRARY_PATH
export PATH=$PATH:/home/tdn5879/codes/hoomd-blue/install/hoomd
export RSEED=$RANDOM
export MERMER=4.0
export RADIUS=2.5
export NOTE='small_'
export MER_TEMP=1.8
export TEMP_TEMP=2.5
export N_HEX1=4
export N_HEX2=2
export N_SCAF=2
export N_pent=1


# run HOOMD with 2 MPI ranks (as requested by #SBATCH -n) each with 1 GPU (2 GPUs in total)

mpirun python main.py

