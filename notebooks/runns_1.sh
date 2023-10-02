#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo Sampling with dynesty
python sampledynesty.py --config_ns=config.py:desyr1 --config_ns.sampler=ns --config_ns.use_emu=False --config_ns.dynesty.nlive=1500 --config_ns.samplername=2

# echo Sampling with EMCEE
# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=1 --config.use_emu=False --config.samplername=testing
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

# addqueue -q cmbgpu -n 1x4 -m 5 -s ./runns_1.sh