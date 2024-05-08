import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import jax
import jax.numpy as jnp
from lsst.functions import load_data, jit_theory, get_params_vec
import jax_cosmo as jc

jc.power.USE_EMU = False


def emcee_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
):
    theory = jit_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    diff = data - theory
    chi2 = diff @ precision @ diff
    return -0.5 * jnp.min(jnp.array([chi2, 1e32]))


@jax.jit
def jit_emcee_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
):
    return emcee_loglike(
        parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )


@jax.jit
def jit_emcee_grad_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
):
    return jax.jacrev(emcee_loglike)(
        parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )


data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl = load_data(
    path="data/lsst_mock_data.fits"
)


cosmo = jc.Cosmology(
    sigma8=0.82,
    Omega_c=0.265,
    Omega_b=0.045,
    h=0.7,
    n_s=0.965,
    w0=-1.0,
    Omega_k=0.0,
    wa=0.0,
)

parameters = get_params_vec(
    cosmo,
    [1e-3] * 5,  # multiplicative
    [1e-3] * 5,  # delta shear
    [0.0, 0.0],  # ia
    [
        1.376695,
        1.451179,
        1.528404,
        1.607983,
        1.689579,
        1.772899,
        1.857700,
        1.943754,
        2.030887,
        2.118943,
    ],  # bias
    [1e-3] * 10,
)  # delta galaxy


loglike = jit_emcee_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
)

grad_loglike = jit_emcee_grad_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
)

# time these two
loglike = jit_emcee_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
)

grad_loglike = jit_emcee_grad_loglike(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
)
