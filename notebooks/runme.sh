#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo
python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=500 --config.barker.nsamples=5000 --config.barker.nchain=2 --config.barker.chainmethod=vectorized --config.use_emu=False --config.samplername=1