import os
import pickle
import jax
import emcee
import sacc
import jax_cosmo as jc
import jax.numpy as jnp
from absl import flags, app
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import ConfigDict

# numpyro
from numpyro.infer import MCMC, HMC, NUTS, BarkerMH, init_to_median
from numpyro.handlers import seed
from numpyro.infer import Predictive, SVI, Trace_ELBO
import numpyro.infer.autoguide as autoguide

# our script
from cosmology.samplenumpyro import numpyro_model
from cosmology.bandpowers import get_nz, scale_cuts, extract_bandwindow, extract_data_covariance, get_params_vec
from cosmology.sampleemcee import jit_theory, emcee_logpost

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.")

def get_test_param():
    cosmo = jc.Cosmology(sigma8=0.852, Omega_c = 0.239, Omega_b=0.054,
                        h = 0.653, n_s=0.933, w0=-1., Omega_k=0., wa=0.)

    parameter = get_params_vec(cosmo,
                        [0.074, 0.186, -0.075, -0.108],
                        [-0.008, -0.100, -0.0018, -0.0097],
                        [0.359, -0.011],
                        [1.34, 1.58, 1.60, 1.90, 1.94],
                        [0.022, -0.0015, 0.02, 0.0097, -0.012])
    return parameter


def load_data(fname = 'cls_DESY1', kmax = 0.15, lmin_wl = 30, lmax_wl = 2000):
    saccfile = sacc.Sacc.load_fits(f"data/{fname}.fits")
    jax_nz_wl = get_nz(saccfile, tracertype="wl")
    jax_nz_gc = get_nz(saccfile, tracertype="gc")
    saccfile_cut = scale_cuts(saccfile, kmax = kmax, lmin_wl = lmin_wl, lmax_wl = lmax_wl)
    bw_gc, bw_gc_wl, bw_wl = extract_bandwindow(saccfile_cut)
    data, covariance = extract_data_covariance(saccfile_cut)
    precision = jnp.linalg.inv(covariance)
    newcov = covariance + jnp.eye(data.shape[0]) * 1E-18
    newprec = jnp.linalg.inv(newcov)
    return data, newprec, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl


def sampling_nuts(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl, cfg):
    nuts_kernel = NUTS(numpyro_model, step_size=cfg.nuts.stepsize, init_strategy=init_to_median, dense_mass=cfg.nuts.dense_mass, max_tree_depth=cfg.nuts.max_tree_depth)
    mcmc_nuts = MCMC(nuts_kernel, num_warmup=cfg.nuts.nwarmup, num_samples=cfg.nuts.nsamples, num_chains=cfg.nuts.nchain,
                chain_method=cfg.nuts.chainmethod,
                progress_bar=True)
    mcmc_nuts.run(jax.random.PRNGKey(cfg.nuts.rng), data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    return mcmc_nuts

def sampling_barker(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl, cfg):
    barker_kernel = BarkerMH(numpyro_model, step_size=cfg.barker.stepsize, init_strategy=init_to_median, dense_mass=cfg.barker.dense_mass)
    mcmc_barker = MCMC(barker_kernel, num_warmup=cfg.barker.nwarmup, num_samples=cfg.barker.nsamples, num_chains=cfg.barker.nchain,
                chain_method=cfg.barker.chainmethod, progress_bar=True)
    mcmc_barker.run(jax.random.PRNGKey(cfg.barker.rng), data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    return mcmc_barker

def sampling_emcee(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl, cfg):
    parameter = get_test_param()
    test_theory = jit_theory(parameter, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    nparams = len(parameter)
    pos = parameter + cfg.emcee.eps * jax.random.normal(jax.random.PRNGKey(0), (2*nparams, nparams))
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_logpost, args=(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl))
    sampler.run_mcmc(pos, cfg.emcee.nsamples, progress=True)
    return sampler


def main(_):
    """
    Run the main sampling code and stores the samples.
    """
    cfg = FLAGS.config
    jc.power.USE_EMU = cfg.use_emu

    # run the sampler
    data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl = load_data(fname = 'cls_DESY1', kmax = 0.15, lmin_wl = 30, lmax_wl = 2000)

    if cfg.sampler == 'nuts':
        mcmc_nuts = sampling_nuts(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl, cfg)
    elif cfg.sampler == 'barker':
        mcmc_barker = sampling_barker(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl, cfg)
    else:
        mcmc_emcee = sampling_emcee(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl, cfg)

if __name__ == "__main__":
    app.run(main)


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl = load_data(fname = 'cls_DESY1', kmax = 0.15, lmin_wl = 30, lmax_wl = 2000)

# with seed(rng_seed=42):
#     theory, logp = numpyro_model(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)

# # use emulator or not
# jc.power.USE_EMU = False
# NWARMUP = 20
# NSAMPLES = 5

# mcmc_nuts = sampling_nuts(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
# mcmc_barker = sampling_barker(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
# mcmc_emcee = sampling_emcee(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
