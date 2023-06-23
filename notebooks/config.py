from ml_collections.config_dict import ConfigDict


def get_config(experiment) -> ConfigDict:
    config = ConfigDict()
    config.logname = "des-analysis"
    config.experiment = experiment

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

    return config
