#!/bin/bash
export COBAYA_USE_FILE_LOCKING=false
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
time python samplecobaya.py

# conda activate jaxcosmo
# /usr/local/shared/slurm/bin/srun -N 2 -n 2 --ntasks-per-node 3 -m cyclic --mpi=pmi2 /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo/bin/python samplecobaya.py
# addqueue -n 2x4 -s -q cmb -c cobaya_test -m 1 ./runcobaya.sh
# addqueue -s -q gpulong -m 10 --gpus 1 ./runcobaya.sh
# addqueue -s -q gpulong -m 10 --gpus 1 --gputype rtx2080with12gb ./runcobaya.sh
