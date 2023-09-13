import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pyccl as ccl

# our scripts and functions
from cosmology.sampleemcee import emcee_logprior
from cosmology.bandpowers import (
    unpack_params_vec,
    get_bandpowers_gc,
    get_bandpowers_gc_wl,
    get_bandpowers_wl,
)

jax.config.update("jax_enable_x64", True)

ZMAX = 2.0
ELLMIN = 2
NELL = 30
ELLMAX_GC = 300
ELLMAX_WL = 3000
BAD_PS = 1e32


def intrinsic_alignment(redshift, amplitude, eta, redshift_fixed=0.62):
    return amplitude * ((1 + redshift) / (1 + redshift_fixed)) ** eta


def run_simulator(cosmology):
    try:
        ccl_cosmo = ccl.Cosmology(
            Omega_c=cosmology.Omega_c.item(),
            Omega_b=cosmology.Omega_b.item(),
            h=cosmology.h.item(),
            sigma8=cosmology.sigma8.item(),
            n_s=cosmology.n_s.item(),
            transfer_function="boltzmann_class",
        )
    except:
        ccl_cosmo = None

    return ccl_cosmo


class cosmoclass:
    def __init__(self, parameters, jax_nz_wl, jax_nz_gc):
        (
            self.cosmo,
            self.multiplicative,
            self.deltaz_wl,
            (a_ia_param, eta_param),
            self.bias,
            self.deltaz_gc,
        ) = unpack_params_vec(parameters)

        self.a_ia_param = a_ia_param
        self.eta_param = eta_param

        self.nz_wl_sys = [
            jc.redshift.systematic_shift(nzi, dzi)
            for nzi, dzi in zip(jax_nz_wl, self.deltaz_wl)
        ]
        self.nz_gc_sys = [
            jc.redshift.systematic_shift(nzi, dzi, zmax=ZMAX)
            for nzi, dzi in zip(jax_nz_gc, self.deltaz_gc)
        ]
        self.ells_coarse_wl = jnp.geomspace(ELLMIN, ELLMAX_WL, NELL, dtype=jnp.float32)
        self.ells_coarse_gc = jnp.geomspace(ELLMIN, ELLMAX_GC, NELL, dtype=jnp.float32)

        self.nbin_gc = len(self.deltaz_gc)
        self.nbin_wl = len(self.deltaz_wl)

        self.ccl_cosmo = run_simulator(self.cosmo)


def ccl_wl_powerspectra(
    nz_wl_sys, multiplicative, a_ia_param, eta_param, ccl_cosmo, ells_coarse_wl
):
    cl_wl_ccl = []
    nbin_wl = len(multiplicative)

    for i in range(nbin_wl):
        for j in range(i, nbin_wl):
            z_i = nz_wl_sys[i].params[0].params[0]
            z_j = nz_wl_sys[j].params[0].params[0]

            nz_i = nz_wl_sys[i].pz_fn(z_i)
            nz_j = nz_wl_sys[j].pz_fn(z_j)

            m_i = multiplicative[i].item()
            m_j = multiplicative[j].item()

            A_IA_i = intrinsic_alignment(z_i, a_ia_param, eta_param)
            A_IA_j = intrinsic_alignment(z_j, a_ia_param, eta_param)

            if ccl_cosmo is not None:
                t1 = ccl.WeakLensingTracer(
                    ccl_cosmo, dndz=(z_i, nz_i), has_shear=True, ia_bias=(z_i, A_IA_i)
                )
                t2 = ccl.WeakLensingTracer(
                    ccl_cosmo, dndz=(z_j, nz_j), has_shear=True, ia_bias=(z_j, A_IA_j)
                )
                cl = (
                    ccl.angular_cl(ccl_cosmo, t1, t2, ells_coarse_wl)
                    * (1.0 + m_i)
                    * (1.0 + m_j)
                )
            else:
                cl = jnp.ones_like(ells_coarse_wl) * BAD_PS

            cl_wl_ccl.append(cl)
    return cl_wl_ccl


def ccl_gc_powerspectra(nz_gc_sys, bias, ccl_cosmo, ells_coarse_gc):
    cl_gc_ccl = []
    nbin_gc = len(bias)
    for i in range(nbin_gc):
        redshift = nz_gc_sys[i].params[0].params[0]
        z_dist = nz_gc_sys[i].pz_fn(redshift)
        bias_i = bias[i].item() * np.ones_like(redshift)

        if ccl_cosmo is not None:
            tracer = ccl.tracers.NumberCountsTracer(
                ccl_cosmo,
                dndz=(redshift, z_dist),
                bias=(redshift, bias_i),
                has_rsd=False,
            )
            cl = ccl.angular_cl(ccl_cosmo, tracer, tracer, ells_coarse_gc)
        else:
            cl = jnp.ones_like(ells_coarse_gc) * BAD_PS

        cl_gc_ccl.append(cl)
    return cl_gc_ccl


def ccl_gc_wl_powerspectra(
    nz_wl_sys,
    nz_gc_sys,
    multiplicative,
    a_ia_param,
    eta_param,
    bias,
    ccl_cosmo,
    ells_coarse_gc,
):
    nbin_wl = len(multiplicative)
    nbin_gc = len(bias)

    cl_gc_wl_ccl = []
    for i in range(nbin_gc):
        for j in range(nbin_wl):
            # galaxy clustering
            z_i = nz_gc_sys[i].params[0].params[0]
            nz_i = nz_gc_sys[i].pz_fn(z_i)
            b_i = bias[i].item() * np.ones_like(z_i)

            # weak lensing
            z_j = nz_wl_sys[j].params[0].params[0]
            nz_j = nz_wl_sys[j].pz_fn(z_j)
            m_j = multiplicative[j].item()
            A_IA_j = intrinsic_alignment(z_j, a_ia_param, eta_param)

            if ccl_cosmo is not None:
                t_i = ccl.tracers.NumberCountsTracer(
                    ccl_cosmo, dndz=(z_i, nz_i), bias=(z_i, b_i), has_rsd=False
                )
                t_j = ccl.WeakLensingTracer(
                    ccl_cosmo, dndz=(z_j, nz_j), has_shear=True, ia_bias=(z_j, A_IA_j)
                )
                cl = ccl.angular_cl(ccl_cosmo, t_i, t_j, ells_coarse_gc) * (1.0 + m_j)
            else:
                cl = jnp.ones_like(ells_coarse_gc) * BAD_PS

            cl_gc_wl_ccl.append(cl)
    return cl_gc_wl_ccl


def ccl_gc_bandpower_calculation(cosmolib, bandwindow_ells, bandwindow_matrix):
    ccl_ps_gc = ccl_gc_powerspectra(
        cosmolib.nz_gc_sys, cosmolib.bias, cosmolib.ccl_cosmo, cosmolib.ells_coarse_gc
    )

    gc_bandpowers = get_bandpowers_gc(
        bandwindow_ells,
        bandwindow_matrix,
        cosmolib.ells_coarse_gc,
        ccl_ps_gc,
        cosmolib.nbin_gc,
    )
    return gc_bandpowers


def ccl_gc_wl_bandpower_calculation(cosmolib, bandwindow_ells, bandwindow_matrix):
    ccl_ps_gc_wl = ccl_gc_wl_powerspectra(
        cosmolib.nz_wl_sys,
        cosmolib.nz_gc_sys,
        cosmolib.multiplicative,
        cosmolib.a_ia_param,
        cosmolib.eta_param,
        cosmolib.bias,
        cosmolib.ccl_cosmo,
        cosmolib.ells_coarse_gc,
    )

    gc_wl_bandpowers = get_bandpowers_gc_wl(
        bandwindow_ells,
        bandwindow_matrix,
        cosmolib.ells_coarse_gc,
        ccl_ps_gc_wl,
        cosmolib.nbin_gc,
        cosmolib.nbin_wl,
    )
    return gc_wl_bandpowers


def ccl_wl_bandpower_calculation(cosmolib, bandwindow_ells, bandwindow_matrix):
    ccl_ps_wl = ccl_wl_powerspectra(
        cosmolib.nz_wl_sys,
        cosmolib.multiplicative,
        cosmolib.a_ia_param,
        cosmolib.eta_param,
        cosmolib.ccl_cosmo,
        cosmolib.ells_coarse_wl,
    )

    wl_bandpowers = get_bandpowers_wl(
        bandwindow_ells,
        bandwindow_matrix,
        cosmolib.ells_coarse_wl,
        ccl_ps_wl,
        cosmolib.nbin_wl,
    )
    return wl_bandpowers


def ccl_get_bandpowers_theory(cosmolib, bw_gc, bw_gc_wl, bw_wl):
    theory_gc = ccl_gc_bandpower_calculation(cosmolib, bw_gc[0], bw_gc[1])
    theory_gc_wl = ccl_gc_wl_bandpower_calculation(cosmolib, bw_gc_wl[0], bw_gc_wl[1])
    theory_wl = ccl_wl_bandpower_calculation(cosmolib, bw_wl[0], bw_wl[1])

    concat_theory_gc = jnp.concatenate(theory_gc)
    concat_theory_gc_wl = jnp.concatenate(theory_gc_wl)
    concat_theory_wl = jnp.concatenate(theory_wl)
    return jnp.concatenate([concat_theory_gc, concat_theory_gc_wl, concat_theory_wl])


def ccl_likelihood(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
):
    logprior = emcee_logprior(parameters)
    if logprior == -1e32:
        chi_square = 1e32
    else:
        cosmolib = cosmoclass(parameters, jax_nz_wl, jax_nz_gc)
        ccl_theory = ccl_get_bandpowers_theory(cosmolib, bw_gc, bw_gc_wl, bw_wl)
        difference = data - ccl_theory
        chi_square = difference @ precision @ difference
        if not jnp.isfinite(chi_square):
            chi_square = 1e32
    return -0.5 * chi_square


def ccl_logpost(
    parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
):
    loglike = ccl_likelihood(
        parameters, data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl
    )
    logprior = emcee_logprior(parameters)
    return loglike + logprior
