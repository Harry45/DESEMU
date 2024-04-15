PROPOSAL = 1e-3

params = {
    # cosmological parameters
    "sigma8": {
        "prior": {"min": 0.60, "max": 1.0},
        "ref": {"dist": "norm", "loc": 0.82, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "omegac": {
        "prior": {"min": 0.14, "max": 0.35},
        "ref": {"dist": "norm", "loc": 0.265, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "omegab": {
        "prior": {"min": 0.03, "max": 0.055},
        "ref": {"dist": "norm", "loc": 0.045, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "hubble": {
        "prior": {"min": 0.64, "max": 0.82},
        "ref": {"dist": "norm", "loc": 0.70, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "ns": {
        "prior": {"min": 0.87, "max": 1.07},
        "ref": {"dist": "norm", "loc": 0.965, "scale": 0.01},
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
    "m5": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    # shifts weak lensing bins
    "dz_wl_1": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "proposal": PROPOSAL,
    },
    "dz_wl_2": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "proposal": PROPOSAL,
    },
    "dz_wl_3": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "proposal": PROPOSAL,
    },
    "dz_wl_4": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "proposal": PROPOSAL,
    },
    "dz_wl_5": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.02},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.02},
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
        "ref": {"dist": "norm", "loc": 1.376695, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b2": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.451179, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b3": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.528404, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b4": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.607983, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b5": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.689579, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b6": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.772899, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b7": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.857700, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b8": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.943754, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b9": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 2.030887, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b10": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 2.118943, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    # shifts (galaxy clustering)
    "dz_gc_1": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_2": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_3": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_4": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_5": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_6": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_7": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_8": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_9": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "dz_gc_10": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
}
