# Integrated Emulator in JAX-COSMO - a DES analysis
We integrate an emulator of the linear matter power spectrum in JAX-COSMO and we test the performance of difference samplers:
1. EMCEE
2. NUTS
3. Barker MH
4. Nested Sampling (dynesty)

We also have a baseline where we compute the power spectra using `ccl` and sample the posterior with EMCEE.

## Different Samplers

For running the nuts and barker:

```
python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=100 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=False --config.samplername=1
```

For running the emcee sampler:

```
python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10 --config.use_emu=False --config.samplername=1
```

For running `dynesty`:

```
python sampledynesty.py --config_ns=config.py:desyr1 --config.sampler=ns --config.use_emu=False --config_ns.dynesty.nlive=100 --config.samplername=1
```

## CCL and EMCEE
For running the EMCEE sampler with CCL:

```
python sampleccl.py --configccl=config.py:desyr1 --configccl.ccl.nsamples=2 --configccl.samplername=test
```