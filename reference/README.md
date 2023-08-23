### Installing Cobaya

```
python -m pip install cobaya --upgrade
python -c "import cobaya"
```

The second line should work without any issue. To install a set of cosmological tool in a specified folder:

```
cobaya-install cosmo -p cl_like/
```

### How to use Cobaya

- Go to the `xCell-likelihoods/ClLike/` folder and

```
python -m pip install .
```

- To minimize the log-likelihood:

```
cobaya-run desy1_3x2pt.yml --minimize
```

- To sample the posterior:

```
cobaya-run desy1_3x2pt.yml
```