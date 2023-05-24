"""
Author: Arrykrishna
Date: May 2023
Email: arrykrish@gmail.com
Project: Inference of bias parameters.
Script: The main configuration file
"""
from ml_collections.config_dict import ConfigDict


def get_config(experiment: str) -> ConfigDict:
    config = ConfigDict()
    config.logname = "emulator-" + experiment
    config.experiment = experiment

    config.value = 10
    return config
