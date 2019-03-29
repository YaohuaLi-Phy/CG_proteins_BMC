#!/bin/bash
#MSUB -N "pmf-np"
#MSUB -l nodes=1:ppn=1:gpus=1
#MSUB -l feature=p100
#MSUB -A b1030
#MSUB -q buyin
#MSUB -l walltime=4:00:00

module load python
module load cuda/cuda_8.0.61_mpich
module load ruby

export RSEED=$RANDOM
export MERMER=5.0
export MER_TEMP=4.0

cd $PBS_O_WORKDIR

mpirun -n 1 python main.py 

