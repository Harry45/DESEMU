# numpyro
import numpyro
import numpyro.distributions as dist
import jax_cosmo as jc

# our scripts
from cosmology.bandpowers import get_params_vec, get_bandpowers_theory


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
    multiplicative = [m1, m2, m3, m4]

    # shifts (weak lensing)
    dz_wl_1 = numpyro.sample("dz_wl_1", dist.Normal(-0.001, 0.016))
    dz_wl_2 = numpyro.sample("dz_wl_2", dist.Normal(-0.019, 0.013))
    dz_wl_3 = numpyro.sample("dz_wl_3", dist.Normal(0.009, 0.011))
    dz_wl_4 = numpyro.sample("dz_wl_4", dist.Normal(-0.018, 0.022))
    dz_wl = [dz_wl_1, dz_wl_2, dz_wl_3, dz_wl_4]
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
    bias = [b1, b2, b3, b4, b5]

    # shifts (galaxy clustering)
    dz_gc_1 = numpyro.sample("dz_gc_1", dist.Normal(0.0, 0.007))
    dz_gc_2 = numpyro.sample("dz_gc_2", dist.Normal(0.0, 0.007))
    dz_gc_3 = numpyro.sample("dz_gc_3", dist.Normal(0.0, 0.006))
    dz_gc_4 = numpyro.sample("dz_gc_4", dist.Normal(0.0, 0.01))
    dz_gc_5 = numpyro.sample("dz_gc_5", dist.Normal(0.0, 0.01))
    dz_gc = [dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5]

    parameters = get_params_vec(cosmo, multiplicative, dz_wl, ia_params, bias, dz_gc)
    theory = get_bandpowers_theory(
        parameters, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )
    sampling_distribution = dist.MultivariateNormal(theory, precision_matrix=precision)
    theory_sample = numpyro.sample("y", sampling_distribution, obs=data)
    log_prob = sampling_distribution.log_prob(theory_sample)
    return theory_sample, log_prob
