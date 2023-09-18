from absl import flags, app
from ml_collections.config_flags import config_flags

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax.scipy.stats.norm as jxnorm
from dynesty import NestedSampler

# our script
from cosmology.bandpowers import get_bandpowers_theory, get_params_vec
from sample import load_data
from utils.helpers import save_sampler

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config_ns", None, "Main configuration file.")

data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl = load_data(
    fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000
)


@jax.jit
def jit_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    return get_bandpowers_theory(
        parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )


def get_test_param():
    cosmo = jc.Cosmology(
        sigma8=0.852,
        Omega_c=0.239,
        Omega_b=0.054,
        h=0.653,
        n_s=0.933,
        w0=-1.0,
        Omega_k=0.0,
        wa=0.0,
    )

    testparameter = get_params_vec(
        cosmo,
        [0.074, 0.186, -0.075, -0.108],
        [-0.008, -0.100, -0.0018, -0.0097],
        [0.359, -0.011],
        [1.34, 1.58, 1.60, 1.90, 1.94],
        [0.022, -0.0015, 0.02, 0.0097, -0.012],
    )
    return testparameter


def unit_transform(random_number, xmin, xmax):
    return xmin + random_number * (xmax - xmin)


def dynesty_prior(unit_cube):
    parameter = jnp.array(unit_cube)

    # cosmology
    parameter = parameter.at[0].set(unit_transform(parameter[0], 0.60, 1.0))
    parameter = parameter.at[1].set(unit_transform(parameter[1], 0.14, 0.35))
    parameter = parameter.at[2].set(unit_transform(parameter[2], 0.03, 0.055))
    parameter = parameter.at[3].set(unit_transform(parameter[3], 0.64, 0.82))
    parameter = parameter.at[4].set(unit_transform(parameter[4], 0.87, 1.07))

    # multiplicative
    parameter = parameter.at[5].set(jxnorm.ppf(parameter[5], 0.012, 0.023))
    parameter = parameter.at[6].set(jxnorm.ppf(parameter[6], 0.012, 0.023))
    parameter = parameter.at[7].set(jxnorm.ppf(parameter[7], 0.012, 0.023))
    parameter = parameter.at[8].set(jxnorm.ppf(parameter[8], 0.012, 0.023))

    # shifts
    parameter = parameter.at[9].set(jxnorm.ppf(parameter[9], -0.001, 0.016))
    parameter = parameter.at[10].set(jxnorm.ppf(parameter[10], -0.019, 0.013))
    parameter = parameter.at[11].set(jxnorm.ppf(parameter[11], 0.009, 0.011))
    parameter = parameter.at[12].set(jxnorm.ppf(parameter[12], -0.018, 0.022))

    # intrinsic alignment
    parameter = parameter.at[13].set(unit_transform(parameter[13], -1.0, 1.0))
    parameter = parameter.at[14].set(unit_transform(parameter[14], -5.0, 5.0))

    # multiplicative bias (galaxy clustering)
    parameter = parameter.at[15].set(unit_transform(parameter[15], 0.8, 3.0))
    parameter = parameter.at[16].set(unit_transform(parameter[16], 0.8, 3.0))
    parameter = parameter.at[17].set(unit_transform(parameter[17], 0.8, 3.0))
    parameter = parameter.at[18].set(unit_transform(parameter[18], 0.8, 3.0))
    parameter = parameter.at[19].set(unit_transform(parameter[19], 0.8, 3.0))

    # shifts (galaxy clustering)
    parameter = parameter.at[20].set(jxnorm.ppf(parameter[20], 0.0, 0.007))
    parameter = parameter.at[21].set(jxnorm.ppf(parameter[21], 0.0, 0.007))
    parameter = parameter.at[22].set(jxnorm.ppf(parameter[22], 0.0, 0.006))
    parameter = parameter.at[23].set(jxnorm.ppf(parameter[23], 0.0, 0.01))
    parameter = parameter.at[24].set(jxnorm.ppf(parameter[24], 0.0, 0.01))
    return parameter


def dynesty_loglike(parameters):
    theory = jit_theory(parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    diff = data - theory
    chi2 = diff @ precision @ diff
    isnan = jnp.isnan(chi2)
    chi2 = jnp.where(isnan, 1e32, chi2)
    return -0.5 * chi2


def main(_):
    cfg = FLAGS.config_ns
    parameter = get_test_param()
    print(cfg.dynesty.nlive, cfg.dynesty.ndim)
    test_theory = jit_theory(parameter, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)
    des_sampler = NestedSampler(
        dynesty_loglike, dynesty_prior, ndim=cfg.dynesty.ndim, nlive=cfg.dynesty.nlive
    )
    des_sampler.run_nested()
    save_sampler(des_sampler, cfg)


if __name__ == "__main__":
    app.run(main)
