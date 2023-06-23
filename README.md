# DESEMU
We develop a code to emulate the 3D matter power spectrum and couple it to the DES weak lensing-galaxy clustering data. This data has been processed by:
- Carlos
- David

and other colleagues at the university. In brief, once the scale cuts are applied, it consists of the following:
```
Galaxy-Galaxy: 0 - 5 data points
Galaxy-Galaxy: 1 - 8 data points
Galaxy-Galaxy: 2 - 10 data points
Galaxy-Galaxy: 3 - 11 data points
Galaxy-Galaxy: 4 - 13 data points
------------------------------------
Galaxy-Shear: (0, 0) - 5 data points
Galaxy-Shear: (0, 1) - 5 data points
Galaxy-Shear: (0, 2) - 5 data points
Galaxy-Shear: (0, 3) - 5 data points
Galaxy-Shear: (1, 0) - 8 data points
Galaxy-Shear: (1, 1) - 8 data points
Galaxy-Shear: (1, 2) - 8 data points
Galaxy-Shear: (1, 3) - 8 data points
Galaxy-Shear: (2, 0) - 10 data points
Galaxy-Shear: (2, 1) - 10 data points
Galaxy-Shear: (2, 2) - 10 data points
Galaxy-Shear: (2, 3) - 10 data points
Galaxy-Shear: (3, 0) - 11 data points
Galaxy-Shear: (3, 1) - 11 data points
Galaxy-Shear: (3, 2) - 11 data points
Galaxy-Shear: (3, 3) - 11 data points
Galaxy-Shear: (4, 0) - 13 data points
Galaxy-Shear: (4, 1) - 13 data points
Galaxy-Shear: (4, 2) - 13 data points
Galaxy-Shear: (4, 3) - 13 data points
-------------------------------------
Shear-Shear: (0, 0) - 39 data points
Shear-Shear: (0, 1) - 39 data points
Shear-Shear: (0, 2) - 39 data points
Shear-Shear: (0, 3) - 39 data points
Shear-Shear: (1, 1) - 39 data points
Shear-Shear: (1, 2) - 39 data points
Shear-Shear: (1, 3) - 39 data points
Shear-Shear: (2, 2) - 39 data points
Shear-Shear: (2, 3) - 39 data points
Shear-Shear: (3, 3) - 39 data points
```
resulting in a $625$ dimensional data vector and a $625\times625$ data covariance matrix. The following resources have been used to develop this code:
- [data](https://github.com/xC-ell/growth-history/tree/main), developed by Carlos
- [montepython likelihood](https://github.com/carlosggarcia/montepython_public/tree/emilio/montepython/likelihoods/cl_cross_corr_v3), developed by Carlos
- [sacc](https://github.com/LSSTDESC/sacc/tree/master), LSST collaboration


where the first item consists of the procedures behind the data processing steps.

### Libraries install so far
```
pip install sacc
pip install jaxlib
pip install "jax[cpu]"
pip install ml-collections
pip install swig
pip install camb
pip install pyccl
pip install torch torchvision torchaudio
```

### To Do
- GP emulator
- Blackjax

### How to run code
Example of how the final code will be run.

```
python main.py --config=config.py:experiment --config.value=25
```