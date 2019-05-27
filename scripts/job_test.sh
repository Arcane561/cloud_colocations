#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-13 -p hebbe
#SBATCH -C GPU
#SBATCH -n 10
#SBATCH -t 0-00:10:00

module load Anaconda3
source activate cloud_colocations

cd /c3se/hebbe/users/simonpf/cloud_colocations/scripts
python print_available_gpus.py
