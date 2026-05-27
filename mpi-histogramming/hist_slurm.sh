#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=DavesJob
#SBATCH --mem=100G
#SBATCH --partition=courses

$SRUN mpirun ~/homework4/hist_mpi
