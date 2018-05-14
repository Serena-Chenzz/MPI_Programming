#!/bin/bash
#SBATCH -p physical
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module load Python/3.4.3-goolf-2015a
time mpiexec python test_big.py
