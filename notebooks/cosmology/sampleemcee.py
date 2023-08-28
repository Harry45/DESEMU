import jax
import jax.numpy as jnp

# our script
from cosmology.bandpowers import get_bandpowers_theory

@jax.jit
def jit_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    return get_bandpowers_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)

def emcee_loglike(parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    theory = jit_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    diff = data - theory
    chi2 = diff @ precision @ diff
    if not jnp.isfinite(chi2):
        chi2 = 1E32
    return -0.5*chi2

def emcee_logprior(parameters):

    # cosmological parameters
    logp_sigma8 = jax.scipy.stats.uniform.logpdf(parameters[0], 0.60, 0.40)
    logp_omegac = jax.scipy.stats.uniform.logpdf(parameters[1], 0.14, 0.21)
    logp_omegab = jax.scipy.stats.uniform.logpdf(parameters[2], 0.03, 0.025)
    logp_hubble = jax.scipy.stats.uniform.logpdf(parameters[3], 0.64, 0.18)
    logp_ns = jax.scipy.stats.uniform.logpdf(parameters[4], 0.87, 0.20)

    # multiplicative factor (weak lensing)
    logp_m1 = jax.scipy.stats.norm.logpdf(parameters[5], 0.012, 0.023)
    logp_m2 = jax.scipy.stats.norm.logpdf(parameters[6], 0.012, 0.023)
    logp_m3 = jax.scipy.stats.norm.logpdf(parameters[7], 0.012, 0.023)
    logp_m4 = jax.scipy.stats.norm.logpdf(parameters[8], 0.012, 0.023)

    # shifts (weak lensing)
    logp_dz_wl_1 = jax.scipy.stats.norm.logpdf(parameters[9], -0.001, 0.016)
    logp_dz_wl_2 = jax.scipy.stats.norm.logpdf(parameters[10], -0.019, 0.013)
    logp_dz_wl_3 = jax.scipy.stats.norm.logpdf(parameters[11], 0.009, 0.011)
    logp_dz_wl_4 = jax.scipy.stats.norm.logpdf(parameters[12], -0.018, 0.022)

    # intrinsic alignment
    logp_a_ia = jax.scipy.stats.uniform.logpdf(parameters[13], -1, 2)
    logp_eta = jax.scipy.stats.uniform.logpdf(parameters[14], -5, 10)

    # multiplicative bias (galaxy clustering)
    logp_b1 = jax.scipy.stats.uniform.logpdf(parameters[15], 0.8, 2.2)
    logp_b2 = jax.scipy.stats.uniform.logpdf(parameters[16], 0.8, 2.2)
    logp_b3 = jax.scipy.stats.uniform.logpdf(parameters[17], 0.8, 2.2)
    logp_b4 = jax.scipy.stats.uniform.logpdf(parameters[18], 0.8, 2.2)
    logp_b5 = jax.scipy.stats.uniform.logpdf(parameters[19], 0.8, 2.2)

    # shifts (galaxy clustering)
    logp_dz_gc_1 = jax.scipy.stats.norm.logpdf(parameters[20], 0.0, 0.007)
    logp_dz_gc_2 = jax.scipy.stats.norm.logpdf(parameters[21], 0.0, 0.007)
    logp_dz_gc_3 = jax.scipy.stats.norm.logpdf(parameters[22], 0.0, 0.006)
    logp_dz_gc_4 = jax.scipy.stats.norm.logpdf(parameters[23], 0.0, 0.01)
    logp_dz_gc_5 = jax.scipy.stats.norm.logpdf(parameters[24], 0.0, 0.01)

    logp_cosmology = logp_sigma8 + logp_omegac + logp_omegab + logp_hubble + logp_ns
    logp_multiplicative = logp_m1 + logp_m2 + logp_m3 + logp_m4
    logp_shifts_wl = logp_dz_wl_1 + logp_dz_wl_2 + logp_dz_wl_3 + logp_dz_wl_4
    logp_intrinsic = logp_a_ia + logp_eta
    logp_bias = logp_b1 + logp_b2 + logp_b3 + logp_b4 + logp_b5
    logp_shifts_gc = logp_dz_gc_1 + logp_dz_gc_2 + logp_dz_gc_3 + logp_dz_gc_4 + logp_dz_gc_5
    logp = logp_cosmology + logp_multiplicative + logp_shifts_wl + logp_intrinsic + logp_bias + logp_shifts_gc
    if not jnp.isfinite(logp):
        logp = -1E32
    return logp

def emcee_logpost(parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    loglike = emcee_loglike(parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    logprior = emcee_logprior(parameters)
    return loglike + logprior