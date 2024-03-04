#!/bin/bash
echo $PWD
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
source activate jaxcosmo
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
python rosenbrock.py

# addqueue -s -q gpulong -m 10 --gpus 1 --gputype rtx2080with12gb ./runrb.sh