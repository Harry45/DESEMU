#!/bin/bash
module load cuda/12.3
echo $PWD
which python
echo $(for i in $(seq 1 50); do printf "-"; done)
python rosenbrock.py

# source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
# source activate jaxcosmo
# addqueue -s -q gpulong -m 10 --gpus 1 --gputype rtx3070with8gb ./runrb.sh