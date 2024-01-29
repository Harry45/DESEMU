import os
import shutil
from cobaya.run import run
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


# setting up cobaya, jaxcosmo and emulator
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jc.power.USE_EMU = True
PROPOSAL = 1e-3
NSAMPLES = 500000
OUTPUT_FOLDER = "/mnt/zfsusers/phys2286/projects/DESEMU/outputcobaya/"
if jc.power.USE_EMU:
    OUTPUT_FOLDER += "emulator_2/"
else:
    OUTPUT_FOLDER += "jaxcosmo_2/"

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
    chi2_out = -0.5 * jnp.nan_to_num(chi2)
    return chi2_out.item()


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
        "prior": {"dist": "norm", "loc": 0.0009, "scale": 0.011},
        "ref": {"dist": "norm", "loc": 0.0009, "scale": 0.001},
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

info["sampler"] = {"mcmc": {"max_samples": NSAMPLES, "Rminus1_stop": 0.01}}
info["output"] = OUTPUT_FOLDER + "output_prefix"

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

# from cobaya.model import get_model

# model = get_model(info)

# for i in range(10):
#     point = dict(
#         zip(
#             model.parameterization.sampled_params(),
#             model.prior.sample(ignore_external=True)[0],
#         )
#     )
#     point_ = {k: round(v, 4) for k, v in point.items()}
#     print(point_)
#     logposterior = model.logposterior(point, as_dict=True)
#     print("Full log-posterior:")
#     print("   logposterior:", logposterior["logpost"])
#     print("   logpriors:", logposterior["logpriors"])
#     print("   loglikelihoods:", logposterior["loglikes"])
#     print("   derived params:", logposterior["derived"])
#     print("-" * 100)
