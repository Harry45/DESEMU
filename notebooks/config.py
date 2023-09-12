from ml_collections.config_dict import ConfigDict


def get_config(experiment) -> ConfigDict:
    config = ConfigDict()
    config.logname = "des-analysis"
    config.experiment = experiment
    config.sampler = "nuts"  # 'nuts', 'barker', 'emcee', 'cclemcee'

    # use emulator not (when sampling the posterior), otherwise EH is used.
    config.use_emu = False

    # cosmological parameters
    config.cosmo = cosmo = ConfigDict()
    cosmo.names = ["sigma8", "Omega_cdm", "Omega_b", "h", "n_s"]

    # neutrino settings
    config.neutrino = neutrino = ConfigDict()
    neutrino.N_ncdm = 1.0
    neutrino.deg_ncdm = 3.0
    neutrino.T_ncdm = 0.71611
    neutrino.N_ur = 0.00641
    neutrino.fixed_nm = 0.06

    # CLASS settings
    config.classy = classy = ConfigDict()
    classy.output = "mPk"
    classy.Omega_k = 0.0
    classy.k_max_pk = 50
    classy.k_min_pk = 1e-4
    classy.z_max_pk = 3.0
    classy.nk = 30
    classy.nz = 20

    # priors
    config.priors = {
        "sigma8": {
            "distribution": "uniform",
            "loc": 0.6,
            "scale": 0.4,
            "fiducial": 0.8,
        },
        "Omega_cdm": {
            "distribution": "uniform",
            "loc": 0.07,
            "scale": 0.43,
            "fiducial": 0.2,
        },
        "Omega_b": {
            "distribution": "uniform",
            "loc": 0.03,
            "scale": 0.04,
            "fiducial": 0.04,
        },
        "h": {"distribution": "uniform", "loc": 0.64, "scale": 0.18, "fiducial": 0.7},
        "n_s": {"distribution": "uniform", "loc": 0.87, "scale": 0.2, "fiducial": 1.0},
    }

    # emulator settings
    config.emu = emu = ConfigDict()
    emu.nlhs = 1000
    emu.jitter = 1e-10
    emu.lr = 0.01
    emu.nrestart = 5
    emu.niter = 1000

    # nuts settings
    config.nuts = nuts = ConfigDict()
    nuts.nwarmup = 20
    nuts.nsamples = 5
    nuts.stepsize = 0.1
    nuts.nchain = 1
    nuts.dense_mass = True
    nuts.chainmethod = "parallel"
    nuts.rng = 253
    nuts.max_tree_depth = 5

    # barker settings
    config.barker = barker = ConfigDict()
    barker.nwarmup = 20
    barker.nsamples = 5
    barker.stepsize = 0.01
    barker.nchain = 1
    barker.dense_mass = True
    barker.chainmethod = "sequential"
    barker.rng = 500

    # emcee settings
    config.emcee = emcee = ConfigDict()
    emcee.nsamples = 10
    emcee.rng = 0
    emcee.eps = 1e-4

    # dynesty settings
    config.dynesty = dynesty = ConfigDict()
    dynesty.nlive = 1500
    dynesty.ndim = 25

    # sampling using EMCEE and CCL
    config.ccl = ccl = ConfigDict()
    ccl.eps = 1e-4
    ccl.nsamples = 10
    ccl.rng = 10

    # filename
    config.samplername = "1"

    return config
