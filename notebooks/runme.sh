#!/bin/bash
conda activate jaxcosmo
python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nsamples=10 --config.use_emu=False --config.samplername=1