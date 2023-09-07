import os
import pickle


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


def save_sampler(sampler, cfg):
    if cfg.sampler == "cclemcee":
        fname = f"{cfg.sampler}_{cfg.samplername}"
    else:
        if cfg.use_emu:
            fname = f"emulator_{cfg.sampler}_{cfg.samplername}"

        else:
            fname = f"jaxcosmo_{cfg.sampler}_{cfg.samplername}"
    pickle_save(sampler, "samples", fname)
    pickle_save(cfg, "samples", "config_" + fname)
