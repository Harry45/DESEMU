#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo
python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=100 --config.barker.nsamples=15000 --config.barker.nchain=1 --config.barker.chainmethod=sequential --config.use_emu=True --config.samplername=1