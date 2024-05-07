import os
import shutil
from cobaya.run import run
from cobaya.model import get_model
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import sacc

# our script
from cosmology.bandpowers import get_bandpowers_theory
from cosmology.bandpowers import (
    get_nz,
    scale_cuts,
    extract_bandwindow,
    extract_data_covariance,
)
import jax_cosmo as jc
from utils.helpers import pickle_load

# setting up cobaya, jaxcosmo and emulator
jax.config.update("jax_default_device", jax.devices("cpu")[0])
jc.power.USE_EMU = False
PROPOSAL = 1e-3
NSAMPLES = 500000
MAIN_PATH = "./"  # "/mnt/zfsusers/phys2286/projects/DESEMU/"
OUTPUT_FOLDER = MAIN_PATH + "outputcobaya/withcov/"
if jc.power.USE_EMU:
    OUTPUT_FOLDER += "emulator_1/"
else:
    OUTPUT_FOLDER += "jaxcosmo_1/"

if os.path.exists(OUTPUT_FOLDER) and os.path.isdir(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER)


def load_data(fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000):
    saccfile = sacc.Sacc.load_fits(f"data/{fname}.fits")
    jax_nz_wl = get_nz(saccfile, tracertype="wl")
    jax_nz_gc = get_nz(saccfile, tracertype="gc")
    saccfile_cut = scale_cuts(saccfile, kmax=kmax, lmin_wl=lmin_wl, lmax_wl=lmax_wl)
    print("Loaded data")
    bw_gc, bw_gc_wl, bw_wl = extract_bandwindow(saccfile_cut)
    data, covariance = extract_data_covariance(saccfile_cut)
    newcov = covariance + jnp.eye(data.shape[0]) * 1e-18
    precision = jnp.linalg.inv(newcov)
    return data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl


DATA, PRECISION, NZ_GC, NZ_WL, BW_GC, BW_GC_WL, BW_WL = load_data(
    fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000
)


@jax.jit
def jit_theory(parameters):
    return get_bandpowers_theory(parameters, NZ_GC, NZ_WL, BW_GC, BW_GC_WL, BW_WL)


def cobaya_logl(
    sigma8,
    omegac,
    omegab,
    hubble,
    ns,
    m1,
    m2,
    m3,
    m4,
    dz_wl_1,
    dz_wl_2,
    dz_wl_3,
    dz_wl_4,
    a_ia,
    eta,
    b1,
    b2,
    b3,
    b4,
    b5,
    dz_gc_1,
    dz_gc_2,
    dz_gc_3,
    dz_gc_4,
    dz_gc_5,
):
    parameters = jnp.array(
        [
            sigma8,
            omegac,
            omegab,
            hubble,
            ns,
            m1,
            m2,
            m3,
            m4,
            dz_wl_1,
            dz_wl_2,
            dz_wl_3,
            dz_wl_4,
            a_ia,
            eta,
            b1,
            b2,
            b3,
            b4,
            b5,
            dz_gc_1,
            dz_gc_2,
            dz_gc_3,
            dz_gc_4,
            dz_gc_5,
        ]
    )
    theory = jit_theory(parameters)
    diff = DATA - theory
    chi2 = diff @ PRECISION @ diff
    logl = -0.5 * jnp.nan_to_num(chi2, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return logl.item()


# Set up the input
info = {"likelihood": {"my_likelihood": cobaya_logl}}
info["params"] = {
    # cosmological parameters
    "sigma8": {
        "prior": {"min": 0.60, "max": 1.0},
        "ref": {"dist": "norm", "loc": 0.85, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "omegac": {
        "prior": {"min": 0.14, "max": 0.35},
        "ref": {"dist": "norm", "loc": 0.25, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "omegab": {
        "prior": {"min": 0.03, "max": 0.055},
        "ref": {"dist": "norm", "loc": 0.04, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "hubble": {
        "prior": {"min": 0.64, "max": 0.82},
        "ref": {"dist": "norm", "loc": 0.70, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "ns": {
        "prior": {"min": 0.87, "max": 1.07},
        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    # multiplicative bias parameters
    "m1": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "m2": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "m3": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "m4": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    # shifts weak lensing bins
    "dz_wl_1": {
        "prior": {"dist": "norm", "loc": -0.001, "scale": 0.016},
        "ref": {"dist": "norm", "loc": -0.001, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_wl_2": {
        "prior": {"dist": "norm", "loc": -0.019, "scale": 0.013},
        "ref": {"dist": "norm", "loc": -0.019, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_wl_3": {
        "prior": {"dist": "norm", "loc": 0.009, "scale": 0.011},
        "ref": {"dist": "norm", "loc": 0.009, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_wl_4": {
        "prior": {"dist": "norm", "loc": -0.018, "scale": 0.022},
        "ref": {"dist": "norm", "loc": -0.018, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    # intrinsic alignment
    "a_ia": {
        "prior": {"min": -1.0, "max": 1.0},
        "ref": 0.0,  # {"dist": "norm", "loc": 0.0, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "eta": {
        "prior": {"min": -5.0, "max": 5.0},
        "ref": 0.0,  # {"dist": "norm", "loc": 0.0, "scale": 0.0001},
        "proposal": PROPOSAL,
    },
    # multiplicative bias (galaxy clustering)
    "b1": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.34, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b2": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.57, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b3": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.59, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b4": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.9, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b5": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.9, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    # shifts (galaxy clustering)
    "dz_gc_1": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.007},
        "ref": {"dist": "norm", "loc": 0.02, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_2": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.007},
        "ref": {"dist": "norm", "loc": -0.0015, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_3": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.006},
        "ref": {"dist": "norm", "loc": 0.02, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_4": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.009, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_5": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": -0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
}

pnames = list(info["params"].keys())

if jc.power.USE_EMU:
    covmat = pickle_load("outputcobaya/testing", "cov_emulator")
else:
    covmat = pickle_load("outputcobaya/testing", "cov_jaxcosmo")
info["sampler"] = {
    "mcmc": {
        "max_samples": NSAMPLES,
        "Rminus1_stop": 0.01,
        "covmat": covmat,
        "covmat_params": pnames,
    }
}
info["output"] = OUTPUT_FOLDER + "des"

# normal Python run
updated_info, sampler = run(info)


## if using MPI
# from mpi4py import MPI
# from cobaya.log import LoggedError

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # Run Cobaya
# success = False
# try:
#     updated_info, sampler = run(info)
#     success = True
# except LoggedError as err:
#     pass

# ## Did it work? (e.g. did not get stuck)
# success = all(comm.allgather(success))

# if not success and rank == 0:
#     print("Sampling failed!")


## if we want to check the model


# def calculate_quantities(point: dict, model) -> float:
#     record = {}
#     logposterior = model.logposterior(point, as_dict=True)
#     record["logpost"] = logposterior["logpost"]
#     record["chi2"] = logposterior["loglikes"]["my_likelihood"] * -2
#     record["loglike"] = logposterior["loglikes"]["my_likelihood"]
#     return record


# model = get_model(info)
# paramnames = model.parameterization.sampled_params()
# samples = np.loadtxt(MAIN_PATH + "outputcobaya/jaxcosmo_2/output_prefix.1.txt")
# samples_infer = samples[:, 2:-4]
# point = dict(zip(paramnames, samples_infer[15640]))
# print(point)
# print(calculate_quantities(point, model))
# print(cobaya_logl(*samples_infer[15640]))

# nmcmc = samples_infer.shape[0]
# record = []
# for i in range(nmcmc):
#     point = dict(zip(paramnames, samples_infer[i]))
#     logposterior = model.logposterior(point, as_dict=True)
#     record.append(
#         {
#             "logpost": logposterior["logpost"],
#             "chi2": logposterior["loglikes"]["my_likelihood"] * -2,
#         }
#     )
#     if divmod(i + 1, 1000)[1] == 0:
#         print(f"{i+1} samples completed!")
# print(i)
# print("Full log-posterior:")
# print("   logposterior:", logposterior["logpost"])
# print("   logpriors:", logposterior["logpriors"])
# print("   loglikelihoods:", logposterior["loglikes"])
# print("   chi2 value:", logposterior["loglikes"]["my_likelihood"] * -2)
# print("   derived params:", logposterior["derived"])
# print("-" * 100)

# testing = pd.DataFrame(record)
# testing.to_csv(OUTPUT_FOLDER + "cobayarun_jc_2.csv")
