#!/bin/bash
# ====================================
#SBATCH --job-name=SVMProt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=16G               # Request the full memory of the node
#SBATCH --time=20:00:00
#SBATCH --account=def-xinwang1
# ====================================

python svm.py
