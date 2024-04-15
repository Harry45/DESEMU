import os
import shutil
from datetime import datetime
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import sacc
from cobaya.run import run
from lsst.parameters import params
from lsst.functions import load_data, jit_theory

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
jc.power.USE_EMU = True
ELLMAX_WL = 3000


def get_output_folder(iteration, mainfolder="CobayaLSST/"):
    if jc.power.USE_EMU:
        folder = mainfolder + f"emulator_{iteration}/"
    else:
        folder = mainfolder + f"jaxcosmo_{iteration}/"
    return folder


class CobayaLSST:

    def __init__(
        self, data, precision, params, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    ):
        self.data = data
        self.precision = precision
        self.params = params
        self.pnames = list(params.keys())
        self._postinit(jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl)

    def _postinit(self, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
        self.jax_nz_gc = jax_nz_gc
        self.jax_nz_wl = jax_nz_wl
        self.bw_gc = bw_gc
        self.bw_gc_wl = bw_gc_wl
        self.bw_wl = bw_wl

    def loglike(self, **kwargs):
        params = jnp.array([kwargs[p] for p in self.pnames])
        theory = jit_theory(
            params,
            self.jax_nz_gc,
            self.jax_nz_wl,
            self.bw_gc,
            self.bw_gc_wl,
            self.bw_wl,
        )
        diff = self.data - theory
        chi2 = diff @ self.precision @ diff
        logl = -0.5 * jnp.nan_to_num(chi2, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)
        return logl.item()

    def run_sampler(self, nsamples, iteration, criterion=1e-2):
        info = {
            "likelihood": {
                "LSSTlike": {"external": self.loglike, "input_params": self.pnames}
            }
        }

        info["params"] = self.params
        info["sampler"] = {"mcmc": {"max_samples": nsamples, "Rminus1_stop": criterion}}

        path = get_output_folder(iteration)
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
        info["output"] = path + "lsst"
        updated_info, sampler = run(info, debug=False)
        return sampler


if __name__ == "__main__":

    NSAMPLES = 500

    data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl = load_data(
        path="data/lsst_mock_data.fits"
    )

    lsstlike = CobayaLSST(
        data, precision, params, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )

    start_time = datetime.now()
    sampler_1 = lsstlike.run_sampler(nsamples=NSAMPLES, iteration=1)
    end_time = datetime.now()
    print(f"Time taken for sampler 1 is : {end_time - start_time}")

    start_time = datetime.now()
    sampler_2 = lsstlike.run_sampler(nsamples=NSAMPLES, iteration=2)
    end_time = datetime.now()
    print(f"Time taken for sampler 1 is : {end_time - start_time}")
