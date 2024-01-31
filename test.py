import os
import gc
import time
import numpy as np
import matplotlib.pylab as plt
import emcee
from getdist import plots, MCSamples
import getdist
import scipy.stats as ss
import pandas as pd

import numpyro
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS, init_to_value
from numpyro.handlers import seed
from numpyro.distributions import constraints
from numpyro.diagnostics import summary
from jax import grad, jit, vmap, jacfwd, jacrev
from utils.helpers import dill_save

jax.config.update("jax_default_device", jax.devices("cpu")[0])

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# settings for GPUs (people are always using the first one)
# GPU_NUMBER = 0
# jax.config.update("jax_default_device", jax.devices()[GPU_NUMBER])

normal_prior = ss.norm(0, 1)
nchain = 2


def rosenbrock(xvalues):
    x_i_plus_one = xvalues[1::2]
    x_i = xvalues[0::2]
    term_1 = 0.001 * (x_i_plus_one - x_i**2) ** 2
    term_2 = (x_i - 1) ** 2
    return sum(term_1 + term_2)


def loglikelihood(xvalues):
    return -rosenbrock(xvalues)


@jax.jit
def jit_loglike(xvalues):
    return loglikelihood(xvalues)


@jax.jit
def jit_grad_loglike(xvalues):
    return jax.jacfwd(loglikelihood)(xvalues)


def logposterior(xvalues):
    # logprior = sum([uniform_prior.logpdf(xvalues[i]) for i in range(ndim)])
    logprior = sum([normal_prior.logpdf(xvalues[i]) for i in range(len(xvalues))])
    if np.isfinite(logprior):
        return logprior + jit_loglike(xvalues)
    return -1e32


def calculate_summary(samples_1, samples_2, nlike, ndecimal=3):
    record = []
    for i in range(samples_1.shape[1]):
        testsamples = np.vstack(([samples_1[:, i], samples_2[:, i]]))
        summary_stats = summary(testsamples)
        summary_stats[f"p_{i}"] = summary_stats.pop("Param:0")
        record.append(summary_stats)

    record_df = []
    for i in range(len(record)):
        record_df.append(
            pd.DataFrame(record[i])
            .round(ndecimal)
            .loc[["r_hat", "n_eff", "mean", "std"]]
        )

    record_df = pd.concat(record_df, axis=1).T
    record_df["n_eff"] /= nlike
    return record_df


def model(ndim):
    xvalues = jnp.zeros(ndim)
    for i in range(ndim):
        # y = numpyro.sample(f"x{i}", dist.Uniform(-delta, delta))
        y = numpyro.sample(f"x{i}", dist.Normal(0, 1))
        xvalues = xvalues.at[i].set(y)
    numpyro.factor("log_prob", jit_loglike(xvalues))
    # numpyro.factor("log_prob", loglikelihood(xvalues))


def run_nuts(stepsize, tree_depth, nwarmup, nsamples_nuts, ndim, nchain=2):
    init_strategy = init_to_value(values={f"x{i}": 1.0 for i in range(ndim)})
    nuts_kernel = NUTS(
        model,
        step_size=stepsize,
        dense_mass=True,
        max_tree_depth=tree_depth,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_chains=nchain,
        num_warmup=nwarmup,
        num_samples=nsamples_nuts,
        chain_method="vectorized",
        # jit_model_args=True,
    )
    mcmc.run(
        jax.random.PRNGKey(0),
        ndim=ndim,
        extra_fields=("potential_energy", "num_steps", "accept_prob"),
    )
    nlike_nuts = mcmc.get_extra_fields()["num_steps"].sum().item()
    return mcmc, nlike_nuts


def process_nuts_chains(mcmc, ndim):
    chains = mcmc.get_samples(group_by_chain=True)
    record = []
    for c in range(nchain):
        samples = np.vstack([np.asarray(chains[f"x{i}"][c]) for i in range(ndim)]).T
        record.append(samples)
    return record


def main(dimension, stepsize, tree_depth, nwarmup, nsamples_nuts):
    stats_nuts = {}
    nlike_nuts_record = {}
    time_nuts = {}

    for d in dimension:
        print(f"Sampling dimensions {d}")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        start_time = time.time()
        mcmc, nlike_nuts = run_nuts(stepsize, tree_depth, nwarmup, nsamples_nuts, d)
        time_nuts[d] = time.time() - start_time

        nuts_grouped = process_nuts_chains(mcmc, d)
        stats_nuts[d] = calculate_summary(nuts_grouped[0], nuts_grouped[1], nlike_nuts)
        nlike_nuts_record[d] = nlike_nuts
        del mcmc

        dill_save(stats_nuts, "rosenbrock", f"stats_nuts_{d}")
        dill_save(nlike_nuts_record, "rosenbrock", f"nlike_nuts_{d}")
        dill_save(time_nuts, "rosenbrock", f"time_nuts_{d}")

        del os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]
        gc.collect()


if __name__ == "__main__":
    dimensions = [160]  # np.arange(1, 7, 1) * 20
    main(dimensions, 0.01, 8, 500, 15000)

# tree_depth = 8
# stepsize = 0.01
# nsamples_nuts = 15000
# nwarmup = 500
