import os

# import sys
import warnings
import pickle
import jax.numpy as jnp
import numpy as np
import sacc
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.handlers import seed

# for JAX Cosmo
# sys.path.append("../reference/jax-cosmo/jax_cosmo-master/")
import jax_cosmo as jc
import jax_cosmo.power as jcp

jcp.USE_EMU = False
warnings.filterwarnings("ignore")
jax.config.update("jax_enable_x64", True)
ZMAX = 2.0


def get_nz_wl(sfile):
    tracers_names = list(sfile.tracers.keys())
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])

    nz_wl = list()
    for i in range(nbin_wl):
        name = f"DESwl__{i}"
        distribution = sfile.tracers[name]
        jaxred = jc.redshift.custom_nz(
            distribution.z.astype("float64"),
            distribution.nz.astype("float64"),
            zmax=ZMAX,
        )
        nz_wl.append(jaxred)

    return nz_wl


def scale_cuts(sfile, lmin_wl=30, lmax_wl=2000):
    # First we remove all B-modes
    sfile.remove_selection(data_type="cl_bb")
    sfile.remove_selection(data_type="cl_be")
    sfile.remove_selection(data_type="cl_eb")
    sfile.remove_selection(data_type="cl_0b")

    tracers_names = list(sfile.tracers.keys())
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])

    # apply scale cut for weak lensing
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            tname_1 = f"DESwl__{i}"
            tname_2 = f"DESwl__{j}"
            sfile.remove_selection(
                data_type="cl_ee", tracers=(tname_1, tname_2), ell__gt=lmax_wl
            )
            sfile.remove_selection(
                data_type="cl_ee", tracers=(tname_1, tname_2), ell__lt=lmin_wl
            )
    return sfile


def get_data_type(tracer_combination):
    if "gc" in tracer_combination[0] and "gc" in tracer_combination[1]:
        dtype = "cl_00"
    elif "gc" in tracer_combination[0] and "wl" in tracer_combination[1]:
        dtype = "cl_0e"
    elif "wl" in tracer_combination[0] and "wl" in tracer_combination[1]:
        dtype = "cl_ee"
    return dtype


def get_ells_bandwindow(sfile, tracer_name_1, tracer_name_2, ellmax=3000):
    dtype = get_data_type((tracer_name_1, tracer_name_2))
    idx = sfile.indices(data_type=dtype, tracers=(tracer_name_1, tracer_name_2))
    window = sfile.get_bandpower_windows(idx)
    fine_ells = window.values
    indices = (fine_ells >= 2) & (fine_ells <= ellmax)
    fine_ells = jnp.asarray(fine_ells[indices], dtype=jnp.float32)
    bandwindow = jnp.asarray(window.weight[indices])
    return fine_ells, bandwindow


def extract_bandwindow(sfile, ellmax=3000):
    tracers_names = list(sfile.tracers.keys())
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])
    record = []

    # shear-shear
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            tracer_name_1 = f"DESwl__{i}"
            tracer_name_2 = f"DESwl__{j}"
            ells, bandwindow = get_ells_bandwindow(
                sfile, tracer_name_1, tracer_name_2, ellmax
            )
            # record[key] = {'ells': ells, 'bandwindow': bandwindow}
            record.append(bandwindow)

    return ells, record


def get_index_pairs(nbin1, nbin2=None, auto=False):
    cl_index = list()
    if nbin2 is not None:
        for i in range(nbin1):
            for j in range(nbin2):
                cl_index.append([i, j + nbin1])
    elif auto:
        for i in range(nbin1):
            cl_index.append([i, i])
    else:
        for i in range(nbin1):
            for j in range(i, nbin1):
                cl_index.append([i, j])
    return cl_index


def get_params_vec(cosmo, multiplicative, deltaz, ia_params):  # , bias, deltaz_gc):
    mparam_1, mparam_2, mparam_3, mparam_4 = multiplicative
    dz1, dz2, dz3, dz4 = deltaz
    a_ia_param, eta_param = ia_params
    return jnp.array(
        [
            cosmo.sigma8,
            cosmo.Omega_c,
            cosmo.Omega_b,
            cosmo.h,
            cosmo.n_s,
            mparam_1,
            mparam_2,
            mparam_3,
            mparam_4,
            dz1,
            dz2,
            dz3,
            dz4,
            a_ia_param,
            eta_param,
        ]
    )


def unpack_params_vec(params):
    cosmo = jc.Cosmology(
        sigma8=params[0],
        Omega_c=params[1],
        Omega_b=params[2],
        h=params[3],
        n_s=params[4],
        w0=-1.0,
        Omega_k=0.0,
        wa=0.0,
    )
    mparam_1, mparam_2, mparam_3, mparam_4 = params[5:9]
    dz1, dz2, dz3, dz4 = params[9:13]
    a_ia_param, eta_param = params[13], params[14]
    return (
        cosmo,
        [mparam_1, mparam_2, mparam_3, mparam_4],
        [dz1, dz2, dz3, dz4],
        [a_ia_param, eta_param],
    )


def get_bandpowers(
    bandwindow_ells, bandwindow_matrix, ells_coarse, powerspectra, nbin_wl
):
    # list to record the band powers
    recordbandpowers = []
    counter = 0
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            cls_wl_interp = jnp.exp(
                jnp.interp(
                    jnp.log(bandwindow_ells),
                    jnp.log(ells_coarse),
                    jnp.log(powerspectra[counter]),
                )
            )
            bandpowers = bandwindow_matrix[counter].T @ cls_wl_interp
            recordbandpowers.append(bandpowers)
            counter += 1
    return recordbandpowers


def wl_bandpower_calculation(parameters, jax_nz_wl, bandwindow_ells, bandwindow_matrix):
    cosmo, multiplicative, deltaz_wl, (a_ia_param, eta_param) = unpack_params_vec(
        parameters
    )
    nbin_wl = len(deltaz_wl)

    # apply all the systematics here (shifts, multiplicative bias, intrinsic alignment)
    nz_wl_sys = [
        jc.redshift.systematic_shift(nzi, dzi, zmax=ZMAX)
        for nzi, dzi in zip(jax_nz_wl, deltaz_wl)
    ]

    b_ia = jc.bias.des_y1_ia_bias(a_ia_param, eta_param, 0.62)
    probes_wl = [
        jc.probes.WeakLensing(
            nz_wl_sys, ia_bias=b_ia, multiplicative_bias=multiplicative
        )
    ]

    # calculate the coarse power spectra for weak lensing
    ells_coarse = jnp.geomspace(2, 3000, 30, dtype=jnp.float32)
    idx_pairs_wl = get_index_pairs(nbin_wl, auto=False)
    ps_wl = jc.angular_cl.angular_cl(
        cosmo, ells_coarse, probes_wl, index_pairs=idx_pairs_wl
    )

    # get the bandpowers
    wl_bandpowers = get_bandpowers(
        bandwindow_ells, bandwindow_matrix, ells_coarse, ps_wl, nbin_wl
    )
    return wl_bandpowers


def pickle_save(file: list, folder: str, fname: str) -> None:
    """Stores a list in a folder.
    Args:
        list_to_store (list): The list to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # use compressed format to store data
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "wb") as dummy:
        pickle.dump(file, dummy)


def pickle_load(folder: str, fname: str):
    """Reads a list from a folder.
    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    Returns:
        Any: the stored file
    """
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "rb") as dummy:
        file = pickle.load(dummy)
    return file


def extract_data_covariance(saccfile):
    tracers_names = list(saccfile.tracers.keys())
    nbin_wl = sum(["DESwl__" in tracers_names[i] for i in range(len(tracers_names))])

    indices = []
    # shear-shear
    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            tracer_name_1 = f"DESwl__{i}"
            tracer_name_2 = f"DESwl__{j}"
            _, _, ind = saccfile.get_ell_cl(
                "cl_ee", tracer_name_1, tracer_name_2, return_cov=False, return_ind=True
            )
            indices += list(ind)

    indices = np.array(indices)
    covariance = saccfile.covariance.covmat[indices][:, indices]
    data = saccfile.mean[indices]
    return jnp.array(data), jnp.array(covariance)


def model(data, covariance, jax_nz_wl, bandwindow_ells, bandwindow_matrix):
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
    dz_wl_1 = numpyro.sample("dz_wl_1", dist.Normal(-0.001, 0.016))  # deltaz_wl
    dz_wl_2 = numpyro.sample("dz_wl_2", dist.Normal(-0.019, 0.013))
    dz_wl_3 = numpyro.sample("dz_wl_3", dist.Normal(0.009, 0.011))
    dz_wl_4 = numpyro.sample("dz_wl_4", dist.Normal(-0.018, 0.022))
    dz_wl = [dz_wl_1, dz_wl_2, dz_wl_3, dz_wl_4]
    nbin_wl = len(dz_wl)

    # intrinsic alignment
    a_ia = numpyro.sample("a_ia", dist.Uniform(-1, 1))
    eta = numpyro.sample("eta", dist.Uniform(-5.0, 5.0))

    nz_wl_sys = [
        jc.redshift.systematic_shift(nzi, dzi, zmax=ZMAX)
        for nzi, dzi in zip(jax_nz_wl, dz_wl)
    ]
    b_ia = jc.bias.des_y1_ia_bias(a_ia, eta, 0.62)
    probes_wl = [
        jc.probes.WeakLensing(
            nz_wl_sys, ia_bias=b_ia, multiplicative_bias=multiplicative
        )
    ]

    # calculate the coarse power spectra for weak lensing
    ells_coarse = jnp.geomspace(2, 3000, 30, dtype=jnp.float32)
    idx_pairs_wl = get_index_pairs(nbin_wl, auto=False)
    ps_wl = jc.angular_cl.angular_cl(
        cosmo, ells_coarse, probes_wl, index_pairs=idx_pairs_wl
    )

    # get the bandpowers
    wl_bandpowers = get_bandpowers(
        bandwindow_ells, bandwindow_matrix, ells_coarse, ps_wl, nbin_wl
    )

    sampling_distribution = dist.MultivariateNormal(
        jnp.concatenate(wl_bandpowers), covariance_matrix=covariance
    )
    theory_sample = numpyro.sample("y", sampling_distribution, obs=data)
    log_prob = sampling_distribution.log_prob(theory_sample)
    return theory_sample, log_prob


if __name__ == "__main__":
    saccfile = sacc.Sacc.load_fits("data/cls_DESY1.fits")
    jax_nz_wl = get_nz_wl(saccfile)
    saccfile_cut = scale_cuts(saccfile, lmin_wl=30, lmax_wl=2000)
    bandwindow_ells, bandwindow_matrix = extract_bandwindow(saccfile_cut, ellmax=3000)
    data, datacov = extract_data_covariance(saccfile_cut)
    with seed(rng_seed=42):
        theory, logp = model(
            data, datacov, jax_nz_wl, bandwindow_ells, bandwindow_matrix
        )

    NWARMUP = 20
    NSAMPLES = 50

    nuts_kernel = NUTS(
        model,
        step_size=0.1,
        init_strategy=init_to_median,
        dense_mass=True,
        max_tree_depth=5,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=NWARMUP,
        num_samples=NSAMPLES,
        num_chains=1,
        progress_bar=True,
    )

    mcmc.run(
        jax.random.PRNGKey(253),
        data,
        datacov,
        jax_nz_wl,
        bandwindow_ells,
        bandwindow_matrix,
    )

    pickle_save(mcmc, "samples_test", "test_mcmc_1")

    testing = pickle_load("samples_test", "test_mcmc_1")
    print(testing.get_samples())
