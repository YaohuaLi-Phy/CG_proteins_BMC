#!/bin/bash
#SBATCH --job-name="bmc"
#SBATCH -A b1030
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 100:00:00
#SBATCH -p buyin
#SBATCH --gres=gpu:k80:1
module purge all
module load cuda/cuda_8.0.61_mpich
module load python

cd $SLURM_SUBMIT_DIR

export LD_LIBRARY_PATH=/software/anaconda2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/tdn5879/codes/hoomd-blue/install/hoomd:$LD_LIBRARY_PATH
export PATH=$PATH:/home/tdn5879/codes/hoomd-blue/install/hoomd
export RSEED=$RANDOM
export MERMER=4.0
export pent_c=0.85
export RADIUS=2.5
export NOTE='A20_twist30'
export twist=30
export angle=20
export MER_TEMP=1.8
export TEMP_TEMP=2.5
export N_HEX1=6
export N_HEX2=0
export N_SCAF=2
export N_pent=1
export lB=0.7
export r0=1.2
export B_factor=1.0

# run HOOMD with 2 MPI ranks (as requested by #SBATCH -n) each with 1 GPU (2 GPUs in total)

mpirun python main.py

