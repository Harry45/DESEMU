import os
import gc
import time
import random 
import numpy as np
import emcee
import scipy.stats as ss
import pandas as pd

import numpyro
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary
from utils.helpers import dill_save

NCHAIN = 2
TREE_DEPTH = 8
STEPSIZE = 0.01
NSAMPLES_NUTS = 15000
NWARMUP = 500
THIN = 10
DISCARD = 100

normal_prior = ss.norm(0, 1)


def rosenbrock(xvalues):
    x_i_plus_one = xvalues[1::2]
    x_i = xvalues[0::2]
    term_1 = 9.0 * (x_i_plus_one - x_i**2) ** 2
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


def single_emcee_run(fiducial, discard, thin, ndim):
    pos = fiducial + 1e-3 * np.random.randn(2 * ndim, ndim)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior)
    nsamples_emcee = int((NSAMPLES_NUTS * thin) / (2 * ndim) + discard)
    sampler.run_mcmc(pos, nsamples_emcee, progress=True)
    return sampler


def run_emcee(fiducial, discard, thin, ndim, nchain=2):
    if nchain > 1:
        record_samples = []
        total_samples = 0
        for chain in range(nchain):
            sampler = single_emcee_run(fiducial, discard, thin, ndim)
            emcee_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
            record_samples.append(emcee_samples)
            total_samples += sampler.flatchain.shape[0]
        return record_samples, total_samples

    sampler = single_emcee_run(fiducial, discard, thin, ndim)
    total_samples = sampler.flatchain.shape[0]
    emcee_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return emcee_samples, total_samples


def logposterior(xvalues):
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
        y = numpyro.sample(f"x{i}", dist.Normal(0, 1))
        xvalues = xvalues.at[i].set(y)
    numpyro.factor("log_prob", jit_loglike(xvalues))


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
    )
    random_integer = random.randint(0, 1000)
    # random_integer = 0 
    mcmc.run(
        jax.random.PRNGKey(random_integer),
        ndim=ndim,
        extra_fields=("potential_energy", "num_steps", "accept_prob"),
    )
    nlike_nuts = mcmc.get_extra_fields()["num_steps"].sum().item()
    return mcmc, nlike_nuts


def process_nuts_chains(mcmc, ndim, nchain):
    chains = mcmc.get_samples(group_by_chain=True)
    record = []
    for c in range(nchain):
        samples = np.vstack([np.asarray(chains[f"x{i}"][c]) for i in range(ndim)]).T
        record.append(samples)
    return record


def main(dimension, stepsize, tree_depth, nwarmup, nsamples_nuts, nrepeat = 5):

    for r in range(nrepeat):
        for d in dimension:
            print(f"Sampling dimensions {d} with EMCEE")

            stats_emcee = {}
            nlike_emcee_record = {}
            time_emcee = {}

            initial = np.ones(d)

            start_time = time.time()
            emcee_samples, nlike_emcee = run_emcee(initial, DISCARD, THIN, d, NCHAIN)
            time_emcee[d] = time.time() - start_time

            stats_emcee[d] = calculate_summary(
                emcee_samples[0], emcee_samples[1], nlike_emcee
            )
            nlike_emcee_record[d] = nlike_emcee
            dill_save(stats_emcee, f"rosenbrock/emcee_{r}", f"stats_emcee_{d}")
            dill_save(nlike_emcee_record, f"rosenbrock/emcee_{r}", f"nlike_emcee_{d}")
            dill_save(time_emcee, f"rosenbrock/emcee_{r}", f"time_emcee_{d}")

            print(f"Sampling dimensions {d} with NUTS")

            stats_nuts = {}
            nlike_nuts_record = {}
            time_nuts = {}

            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

            start_time = time.time()
            mcmc, nlike_nuts = run_nuts(stepsize, tree_depth, nwarmup, nsamples_nuts, d)
            time_nuts[d] = time.time() - start_time

            nuts_grouped = process_nuts_chains(mcmc, d, NCHAIN)
            stats_nuts[d] = calculate_summary(nuts_grouped[0], nuts_grouped[1], nlike_nuts)
            nlike_nuts_record[d] = nlike_nuts

            dill_save(stats_nuts, f"rosenbrock/nuts_{r}", f"stats_nuts_{d}")
            dill_save(nlike_nuts_record, f"rosenbrock/nuts_{r}", f"nlike_nuts_{d}")
            dill_save(time_nuts, f"rosenbrock/nuts_{r}", f"time_nuts_{d}")

            del mcmc
            del os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]
            gc.collect()
            jax.clear_backends()


if __name__ == "__main__":
    dimensions = np.arange(1, 11, 1) * 10 # np.arange(4, 50, 4)
    main(dimensions, STEPSIZE, TREE_DEPTH, NWARMUP, NSAMPLES_NUTS, nrepeat=5)
