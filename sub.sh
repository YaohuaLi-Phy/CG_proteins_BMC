#!/bin/bash
#MSUB -N "pmf-np"
#MSUB -l nodes=1:ppn=1:gpus=1
#MSUB -l feature=p100
#MSUB -A b1030
#MSUB -q buyin
#MSUB -l walltime=8:00:00

module load python
module load cuda/cuda_8.0.61_mpich
module load ruby

export RSEED=$RANDOM
export MERMER=4.0
export MER_TEMP=3.0
export TEMP_TEMP=1.5
export N_HEX1=2
export N_SCAF=2

cd $PBS_O_WORKDIR

mpirun -n 1 python main.py 

