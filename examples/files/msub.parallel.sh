#!/bin/bash
#MSUB -l nodes=4:ppn=28,walltime=60 -d /path/to/code/
mpirun -np 80 python parallel.py
