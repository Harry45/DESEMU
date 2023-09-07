#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo
python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=100 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=True --config.samplername=1