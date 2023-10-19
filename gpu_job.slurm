#!/bin/bash
# ====================================
#SBATCH --job-name=SVMProt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=32G               # Request the full memory of the node
#SBATCH --time=24:00:00
#SBATCH --account=def-xinwang1
# ====================================

python svm_parallel.py
