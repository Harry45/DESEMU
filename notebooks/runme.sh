#!/bin/bash
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo
python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nsamples=10 --config.use_emu=False --config.samplername=1