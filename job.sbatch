#!/bin/bash
#SBATCH --job-name=gpu-hw4
#SBATCH --output=out.txt
#SBATCH --error=err.txt #standard error goes to this file
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:1
module purge
#if your executable was built with CUDA, load the CUDA module:
module load cuda
module load gcc/5.4.0
./hw4 256
./hw4 512
./hw4 1024