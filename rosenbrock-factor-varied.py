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

ndim = 26
nsamples_nuts = 15000
nchain = 2
tree_depth = 8
stepsize = 0.01
nwarmup = 500

thin = 10
discard = 100
nsamples_emcee = int((nsamples_nuts * thin) / (2 * ndim) + discard)
fiducial = np.ones(ndim)
normal_prior = ss.norm(0, 1)


def rosenbrock(xvalues, factor):
    x_i_plus_one = xvalues[1::2]
    x_i = xvalues[0::2]
    term_1 = factor * (x_i_plus_one - x_i**2) ** 2
    term_2 = (x_i - 1) ** 2
    return sum(term_1 + term_2)


def loglikelihood(xvalues, factor):
    return -rosenbrock(xvalues, factor)


@jax.jit
def jit_loglike(xvalues, factor):
    return loglikelihood(xvalues, factor)


@jax.jit
def jit_grad_loglike(xvalues, factor):
    return jax.jacfwd(loglikelihood, factor)(xvalues)


def logposterior(xvalues, factor):
    # logprior = sum([uniform_prior.logpdf(xvalues[i]) for i in range(ndim)])
    logprior = sum([normal_prior.logpdf(xvalues[i]) for i in range(len(xvalues))])
    if np.isfinite(logprior):
        return logprior + jit_loglike(xvalues, factor)
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


def single_emcee_run(fiducial, thin, ndim, factor):
    pos = fiducial + 1e-3 * np.random.randn(2 * ndim, ndim)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, args=(factor,))
    sampler.run_mcmc(pos, nsamples_emcee, progress=True)
    return sampler


def run_emcee(fiducial, discard=discard, thin=thin, ndim=ndim, nchain=2, factor=1e-3):
    if nchain > 1:
        record_samples = []
        total_samples = 0
        for chain in range(nchain):
            sampler = single_emcee_run(fiducial, thin, ndim, factor)
            emcee_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
            # emcee_samples = sampler.flatchain[::thin]
            record_samples.append(emcee_samples)
            total_samples += sampler.flatchain.shape[0]
        return record_samples, total_samples

    sampler = single_emcee_run(fiducial, discard, thin, ndim, factor)
    total_samples = sampler.flatchain.shape[0]
    emcee_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    # emcee_samples = sampler.flatchain[::thin]
    return emcee_samples, total_samples


def model(ndim, factor):
    xvalues = jnp.zeros(ndim)
    for i in range(ndim):
        # y = numpyro.sample(f"x{i}", dist.Uniform(-delta, delta))
        y = numpyro.sample(f"x{i}", dist.Normal(0, 1))
        xvalues = xvalues.at[i].set(y)
    numpyro.factor("log_prob", jit_loglike(xvalues, factor))


def run_nuts(
    stepsize, tree_depth, nwarmup, nsamples_nuts, ndim=ndim, nchain=2, factor=1e-3
):
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
    mcmc.run(
        jax.random.PRNGKey(0),
        ndim=ndim,
        factor=factor,
        extra_fields=("potential_energy", "num_steps", "accept_prob"),
    )
    nlike_nuts = mcmc.get_extra_fields()["num_steps"].sum().item()
    return mcmc, nlike_nuts


def process_nuts_chains(mcmc, ndim=ndim):
    chains = mcmc.get_samples(group_by_chain=True)
    record = []
    for c in range(nchain):
        samples = np.vstack([np.asarray(chains[f"x{i}"][c]) for i in range(ndim)]).T
        record.append(samples)
    return record


if __name__ == "__main__":
    factors = np.geomspace(10, 40, 6, endpoint=True)
    nrepeat = 5
    repetition = []

    for r in range(nrepeat):
        record = np.zeros_like(factors)
        print(f"Doing repetition {r+1}")

        for i, f in enumerate(factors):
            print(f"Running EMCEE and NUTS for factor = {f}")

            # EMCEE
            emcee_samples, nlike_emcee = run_emcee(
                fiducial, discard, thin, ndim, nchain=nchain, factor=f
            )
            stats_emcee = calculate_summary(
                emcee_samples[0], emcee_samples[1], nlike_emcee
            )

            # NUTS
            mcmc, nlike_nuts = run_nuts(
                stepsize, tree_depth, nwarmup, nsamples_nuts, ndim, nchain, factor=f
            )
            nuts_grouped = process_nuts_chains(mcmc, ndim)
            stats_nuts = calculate_summary(nuts_grouped[0], nuts_grouped[1], nlike_nuts)

            # calculate quantity
            record[i] = stats_nuts["n_eff"].mean() / stats_emcee["n_eff"].mean()
        repetition.append(record)

    dill_save(
        {"gamma": repetition, "factors": factors},
        "rosenbrock",
        "d_26_different_factors",
    )
