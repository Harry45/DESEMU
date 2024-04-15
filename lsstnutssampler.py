import os
import random
import jax
import jaxlib
import jax_cosmo as jc
import numpyro
from datetime import datetime
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value
from lsst.functions import get_params_vec, get_bandpowers_theory, load_data
from utils.helpers import pickle_save

jax.config.update("jax_enable_x64", True)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

# settings for GPUs
GPU_NUMBER = 0
jax.config.update("jax_default_device", jax.devices()[GPU_NUMBER])
num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind
numpyro.enable_x64()

jc.power.USE_EMU = True
ELLMAX_WL = 3000

print("-" * 50)
print(f"jax version: {jax.__version__}")
print(f"jaxlib version: {jaxlib.__version__}")
print(f"numpyro version: {numpyro.__version__}")
print(f"Found {num_devices} JAX devices of type {device_type}.")
print(f"We are using {jax.devices()[GPU_NUMBER]}")
print("-" * 50)


def numpyro_model(data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl):
    #  Cosmological params
    sigma8 = numpyro.sample("sigma8", dist.Uniform(0.6, 1.0))
    Omegac = numpyro.sample("Omegac", dist.Uniform(0.14, 0.35))
    Omegab = numpyro.sample("Omegab", dist.Uniform(0.03, 0.055))
    hubble = numpyro.sample("hubble", dist.Uniform(0.64, 0.82))
    ns = numpyro.sample("ns", dist.Uniform(0.87, 1.07))
    cosmo = jc.Cosmology(
        Omega_c=Omegac,
        sigma8=sigma8,
        Omega_b=Omegab,
        h=hubble,
        n_s=ns,
        w0=-1.0,
        Omega_k=0.0,
        wa=0.0,
    )

    # multiplicative factor (weak lensing)
    m1 = numpyro.sample("m1", dist.Normal(0.012, 0.023))
    m2 = numpyro.sample("m2", dist.Normal(0.012, 0.023))
    m3 = numpyro.sample("m3", dist.Normal(0.012, 0.023))
    m4 = numpyro.sample("m4", dist.Normal(0.012, 0.023))
    m5 = numpyro.sample("m5", dist.Normal(0.012, 0.023))
    multiplicative = [m1, m2, m3, m4, m5]

    # shifts (weak lensing)
    dz_wl_1 = numpyro.sample("dz_wl_1", dist.Normal(0.0, 0.02))
    dz_wl_2 = numpyro.sample("dz_wl_2", dist.Normal(0.0, 0.02))
    dz_wl_3 = numpyro.sample("dz_wl_3", dist.Normal(0.0, 0.02))
    dz_wl_4 = numpyro.sample("dz_wl_4", dist.Normal(0.0, 0.02))
    dz_wl_5 = numpyro.sample("dz_wl_5", dist.Normal(0.0, 0.02))
    dz_wl = [dz_wl_1, dz_wl_2, dz_wl_3, dz_wl_4, dz_wl_5]
    nbin_wl = len(dz_wl)

    # intrinsic alignment
    a_ia = numpyro.sample("a_ia", dist.Uniform(-1, 1))
    eta = numpyro.sample("eta", dist.Uniform(-5.0, 5.0))
    ia_params = [a_ia, eta]

    # multiplicative bias (galaxy clustering)
    b1 = numpyro.sample("b1", dist.Uniform(0.8, 3.0))
    b2 = numpyro.sample("b2", dist.Uniform(0.8, 3.0))
    b3 = numpyro.sample("b3", dist.Uniform(0.8, 3.0))
    b4 = numpyro.sample("b4", dist.Uniform(0.8, 3.0))
    b5 = numpyro.sample("b5", dist.Uniform(0.8, 3.0))
    b6 = numpyro.sample("b6", dist.Uniform(0.8, 3.0))
    b7 = numpyro.sample("b7", dist.Uniform(0.8, 3.0))
    b8 = numpyro.sample("b8", dist.Uniform(0.8, 3.0))
    b9 = numpyro.sample("b9", dist.Uniform(0.8, 3.0))
    b10 = numpyro.sample("b10", dist.Uniform(0.8, 3.0))
    bias = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]

    # shifts (galaxy clustering)
    dz_gc_1 = numpyro.sample("dz_gc_1", dist.Normal(0.0, 0.01))
    dz_gc_2 = numpyro.sample("dz_gc_2", dist.Normal(0.0, 0.01))
    dz_gc_3 = numpyro.sample("dz_gc_3", dist.Normal(0.0, 0.01))
    dz_gc_4 = numpyro.sample("dz_gc_4", dist.Normal(0.0, 0.01))
    dz_gc_5 = numpyro.sample("dz_gc_5", dist.Normal(0.0, 0.01))
    dz_gc_6 = numpyro.sample("dz_gc_6", dist.Normal(0.0, 0.01))
    dz_gc_7 = numpyro.sample("dz_gc_7", dist.Normal(0.0, 0.01))
    dz_gc_8 = numpyro.sample("dz_gc_8", dist.Normal(0.0, 0.01))
    dz_gc_9 = numpyro.sample("dz_gc_9", dist.Normal(0.0, 0.01))
    dz_gc_10 = numpyro.sample("dz_gc_10", dist.Normal(0.0, 0.01))
    dz_gc = [
        dz_gc_1,
        dz_gc_2,
        dz_gc_3,
        dz_gc_4,
        dz_gc_5,
        dz_gc_6,
        dz_gc_7,
        dz_gc_8,
        dz_gc_9,
        dz_gc_10,
    ]

    parameters = get_params_vec(cosmo, multiplicative, dz_wl, ia_params, bias, dz_gc)
    theory = get_bandpowers_theory(
        parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )
    sampling_distribution = dist.MultivariateNormal(theory, precision_matrix=precision)
    theory_sample = numpyro.sample("y", sampling_distribution, obs=data)
    log_prob = sampling_distribution.log_prob(theory_sample)
    return theory_sample, log_prob


if __name__ == "__main__":
    STEPSIZE = 0.01
    TREE_DEPTH = 8
    NWARMUP = 5
    NSAMPLES = 5

    data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl = load_data(
        path="data/lsst_mock_data.fits"
    )

    ref_params = {
        "sigma8": 0.82,
        "Omega_c": 0.265,
        "Omega_b": 0.045,
        "ns": 0.965,
        "hubble": 0.7,
        "m1": 1e-3,
        "m2": 1e-3,
        "m3": 1e-3,
        "m4": 1e-3,
        "m5": 1e-3,
        "dz_wl_1": 1e-3,
        "dz_wl_2": 1e-3,
        "dz_wl_3": 1e-3,
        "dz_wl_4": 1e-3,
        "dz_wl_5": 1e-3,
        "a_ia": 1e-3,
        "eta": 1e-3,
        "b1": 1.376695,
        "b2": 1.451179,
        "b3": 1.528404,
        "b4": 1.607983,
        "b5": 1.689579,
        "b6": 1.772899,
        "b7": 1.857700,
        "b8": 1.943754,
        "b9": 2.030887,
        "b10": 2.118943,
        "dz_gc_1": 1e-3,
        "dz_gc_2": 1e-3,
        "dz_gc_3": 1e-3,
        "dz_gc_4": 1e-3,
        "dz_gc_5": 1e-3,
        "dz_gc_6": 1e-3,
        "dz_gc_7": 1e-3,
        "dz_gc_8": 1e-3,
        "dz_gc_9": 1e-3,
        "dz_gc_10": 1e-3,
    }
    nuts_kernel = NUTS(
        numpyro_model,
        step_size=STEPSIZE,
        init_strategy=init_to_value(values=ref_params),  # init_to_median,
        dense_mass=True,
        max_tree_depth=TREE_DEPTH,
    )

    mcmc_nuts = MCMC(
        nuts_kernel,
        num_warmup=NWARMUP,
        num_samples=NSAMPLES,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=True,
    )

    start_time = datetime.now()

    mcmc_nuts.run(
        jax.random.PRNGKey(random.randint(0, 1000)),
        data,
        precision,
        jax_nz_gc,
        jax_nz_wl,
        bw_gc,
        bw_gc_wl,
        bw_wl,
    )
    end_time = datetime.now()
    print(f"Time taken for NUTS sampler is : {end_time - start_time}")

    if jc.power.USE_EMU:
        fname = "nuts_sampler_emulator"
    else:
        fname = "nuts_sampler_jaxcosmo"

    pickle_save(mcmc_nuts, "lsst", fname)
