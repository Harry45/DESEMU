# Integrated Emulator in JAX-COSMO - a DES analysis
We integrate an emulator of the linear matter power spectrum in JAX-COSMO and we test the performance of difference samplers:
1. EMCEE
2. NUTS
3. Barker MH
4. Nested Sampling (dynesty)

For running `dynesty`:

```
python sampledynesty.py --config_ns=config.py:desyr1 --config.sampler=ns --config.use_emu=False --config_ns.dynesty.nlive=100 --config.samplername=1
```

For running the other samplers (nuts, barker, emcee):

```
python sample.py --config=config.py:desyr1 --config.sampler=emcee --config.emcee.nsamples=10 --config.use_emu=False --config.samplername=1
```