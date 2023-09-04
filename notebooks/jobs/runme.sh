#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo
python /mnt/zfsusers/phys2286/projects/DESEMU/notebooks/sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nsamples=50 --config.use_emu=False --config.samplername=1