#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-1 -p hebbe -C GPU
#SBATCH -n 10
#SBATCH -t 0-00:10:00

python print_available_gpus.py
