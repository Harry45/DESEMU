import jax

jax.config.update("jax_platform_name", "cpu")
print(f"The default device being used is {jax.default_backend()}")

import emcee
import numpy as np
from absl import flags, app
from ml_collections.config_flags import config_flags

# our scripts and functions
from utils.helpers import save_sampler
from cosmology.cclbandpowers import (
    ccl_get_nz,
    ccl_load_data,
    ccl_get_test_param,
    ccl_logpost,
)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("configccl", None, "Main configuration file.")


def ccl_sampling_emcee(
    ccl_data,
    ccl_precision,
    ccl_nz_gc,
    ccl_nz_wl,
    ccl_bw_gc,
    ccl_bw_gc_wl,
    ccl_bw_wl,
    cfg,
):
    parameter = ccl_get_test_param()
    nparams = len(parameter)
    pos = parameter + cfg.ccl.eps * np.random.normal(0, 1, (2 * nparams, nparams))
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        ccl_logpost,
        args=(
            ccl_data,
            ccl_precision,
            ccl_nz_gc,
            ccl_nz_wl,
            ccl_bw_gc,
            ccl_bw_gc_wl,
            ccl_bw_wl,
        ),
    )
    sampler.run_mcmc(pos, cfg.ccl.nsamples, progress=True)
    save_sampler(sampler, cfg)
    return sampler


def main(_):
    cfg = FLAGS.configccl
    ccl_nz_wl = ccl_get_nz(fname="cls_DESY1", tracertype="wl")
    ccl_nz_gc = ccl_get_nz(fname="cls_DESY1", tracertype="gc")
    ccl_data, ccl_precision, ccl_bw_gc, ccl_bw_gc_wl, ccl_bw_wl = ccl_load_data(
        fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000
    )
    sampler = ccl_sampling_emcee(
        ccl_data,
        ccl_precision,
        ccl_nz_gc,
        ccl_nz_wl,
        ccl_bw_gc,
        ccl_bw_gc_wl,
        ccl_bw_wl,
        cfg,
    )


if __name__ == "__main__":
    app.run(main)
