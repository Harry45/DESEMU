#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo Sampling with dynesty
# python sampledynesty.py --config_ns=config.py:desyr1 --config_ns.sampler=ns --config_ns.use_emu=True --config_ns.dynesty.nlive=1500 --config_ns.samplername=2
# addqueue -q cmbgpu -n 1x4 -m 5 -s ./runns_1.sh