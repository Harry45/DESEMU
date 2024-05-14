## Emulation and Assessment of Gradient-Based samplers in Cosmology
We integrate an emulator of the linear matter power spectrum in JAX-COSMO and we test the performance of difference samplers:
1. Cobaya
2. NUTS

### Reference Papers
- [arXiv:2302.05163](https://arxiv.org/abs/2302.05163)
- [arXiv:2105.12108](https://arxiv.org/abs/2105.12108)
- [arXiv:2310.08306](https://arxiv.org/abs/2310.08306)

### Running the Samplers

- For running `NUTS`:

```
python sample.py --config=config.py:desyr1 --config.sampler=nuts --config.nuts.nwarmup=100 --config.nuts.nsamples=15000 --config.nuts.nchain=2 --config.nuts.chainmethod=vectorized --config.use_emu=False --config.samplername=1
```

- For running `Cobaya`:

```
python samplecobaya.py
```

but we have to specify the setup (for example, number of samples) in the script first.

- For the Multivariate Normal Distribution and the Rosenbrock function, the notebooks `rosenbrock.ipynb` and `multivariate.ipynb` are used.