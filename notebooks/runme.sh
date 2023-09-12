#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo
python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=30000 --config.use_emu=False --config.samplername=2
# python sample.py --config=config.py:desyr1 --config.sampler=cclemcee --config.ccl.nsamples=5000 --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=5000 --config.use_emu=True --config.samplername=2
