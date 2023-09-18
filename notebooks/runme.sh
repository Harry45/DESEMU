#!/bin/bash
source /mnt/zfsusers/phys2286/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/zfsusers/phys2286/anaconda3/envs/jaxcosmo

## EMCEE
# SECONDS=0
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=False --config.samplername=1
# echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

# SECONDS=0
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=False --config.samplername=2
# echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

# SECONDS=0
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=True --config.samplername=1
# echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

# SECONDS=0
# python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10000 --config.use_emu=True --config.samplername=2
# echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

## Dynesty
SECONDS=0
python sampledynesty.py --config_ns=config.py:desyr1 --config.sampler=ns --config.use_emu=False --config_ns.dynesty.nlive=1500 --config.samplername=1
echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

SECONDS=0
python sampledynesty.py --config_ns=config.py:desyr1 --config.sampler=ns --config.use_emu=False --config_ns.dynesty.nlive=1500 --config.samplername=2
echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

SECONDS=0
python sampledynesty.py --config_ns=config.py:desyr1 --config.sampler=ns --config.use_emu=True --config_ns.dynesty.nlive=1500 --config.samplername=1
echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

SECONDS=0
python sampledynesty.py --config_ns=config.py:desyr1 --config.sampler=ns --config.use_emu=True --config_ns.dynesty.nlive=1500 --config.samplername=2
echo "Elapsed time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

## CCL and EMCEE
# python sampleccl.py --configccl=config.py:desyr1 --configccl.sampler=cclemcee --configccl.ccl.nsamples=10000 --configccl.samplername=camb_1
# python sampleccl.py --configccl=config.py:desyr1 --configccl.sampler=cclemcee --configccl.ccl.nsamples=10000 --configccl.samplername=camb_2

## Barker
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=False --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=False --config.samplername=2
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=True --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=barker --config.barker.nwarmup=1000 --config.barker.nsamples=150000 --config.use_emu=True --config.samplername=2

## NUTS
# python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=100 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=True --config.samplername=1
# python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=100 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=False --config.samplername=1
