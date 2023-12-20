#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate jaxcosmo
which python
echo $(for i in $(seq 1 50); do printf "-"; done)

# echo Sampling with CCL and EMCEE
# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sampleccl.py --configccl=config.py:desyr1 --configccl.sampler=cclemcee --configccl.ccl.nsamples=10000 --configccl.samplername=camb_1
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sampleccl.py --configccl=config.py:desyr1 --configccl.sampler=cclemcee --configccl.ccl.nsamples=10000 --configccl.samplername=camb_2
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

# echo Sampling with EMCEE
# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=False --config.samplername=1
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=False --config.samplername=2
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=True --config.samplername=1
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

# echo $(for i in $(seq 1 100); do printf "-"; done)
# date_start=`date +%s`
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=True --config.samplername=2
# date_end=`date +%s`
# seconds=$((date_end-date_start))
# minutes=$((seconds/60))
# seconds=$((seconds-60*minutes))
# hours=$((minutes/60))
# minutes=$((minutes-60*hours))
# echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
# echo $(for i in $(seq 1 100); do printf "-"; done)

## EMCEE
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=False --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=False --config.samplername=2
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=True --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=True --config.samplername=2

## Dynesty
# python sampledynesty.py --config_ns=config.py:desyr1 --config_ns.sampler=ns --config_ns.use_emu=False --config_ns.dynesty.nlive=1500 --config_ns.samplername=1
# python sampledynesty.py --config_ns=config.py:desyr1 --config_ns.sampler=ns --config_ns.use_emu=False --config_ns.dynesty.nlive=1500 --config_ns.samplername=2
# python sampledynesty.py --config_ns=config.py:desyr1 --config_ns.sampler=ns --config_ns.use_emu=True --config_ns.dynesty.nlive=1500 --config_ns.samplername=1
# python sampledynesty.py --config_ns=config.py:desyr1 --config_ns.sampler=ns --config_ns.use_emu=True --config_ns.dynesty.nlive=1500 --config_ns.samplername=2

## CCL and EMCEE
# python sampleccl.py --configccl=config.py:desyr1 --configccl.sampler=cclemcee --configccl.ccl.nsamples=10000 --configccl.samplername=camb_1
# python sampleccl.py --configccl=config.py:desyr1 --configccl.sampler=cclemcee --configccl.ccl.nsamples=10000 --configccl.samplername=camb_2

## Barker
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=False --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=False --config.samplername=2
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=True --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=True --config.samplername=2

## NUTS
echo Sampling with NUTS with stepsize of 0.01 and tree depth of 7
# python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=1000 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=True --config.samplername=2
python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=1000 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=False --config.samplername=small_ss_high_td

## Submitting Job
# addqueue -q gpulong -n 1x4 -m 5 -s ./runme.sh