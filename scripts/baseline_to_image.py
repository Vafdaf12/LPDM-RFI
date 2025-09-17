import h5py
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def inpaint_clean(file: h5py.File, baseline: str) -> np.ndarray:
    """In-paints an RFI-corrupted baseline using the data predicted by CLEAN"""
    # Data coming from the telescope
    eta = file[f"{baseline}/CORRUPTED_DATA"][:, :, polarization]
    eta = np.abs(eta)

    # CLEAN model predicted from uncorrupted data
    model_data = file[f"{baseline}/MODEL_DATA"][:, :, polarization]
    model_data = np.abs(model_data)

    # In-painted result
    flags = file[f"{baseline}/FLAG"][:, :, polarization]
    eta[flags] =  model_data[flags]
    
    return eta

def inpaint_mean(file: h5py.File, baseline: str) -> np.ndarray:
    """In-paints an RFI-corrupted baseline using the mean value of the unflagged data"""
    # Data coming from the telescope
    eta = file[f"{baseline}/CORRUPTED_DATA"][:, :, polarization]
    eta = np.abs(eta)
    
    # In-painted result
    flags = file[f"{baseline}/FLAG"][:, :, polarization]
    eta[flags] = eta[~flags].mean()
    
    return eta

def inpaint_zero(file: h5py.File, baseline: str) -> np.ndarray:
    """In-paints an RFI-corrupted baseline by zeroing out flagged regions"""
    # Data coming from the telescope
    eta = file[f"{baseline}/CORRUPTED_DATA"][:, :, polarization]
    eta = np.abs(eta)

    # In-painted result
    flags = file[f"{baseline}/FLAG"][:, :, polarization]
    eta[flags] = 0
    
    return eta

@dataclass
class Args:
    dataset: str
    targetdir: str
    method: str

METHODS = {
    "clean": inpaint_clean,
    "mean": inpaint_mean,
    "zero": inpaint_zero,
}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="Baseline to Image Conversion",
        description="In-paint and standardize baselines from an HDF5 dataset.",
    )
    parser.add_argument("dataset", help=".h5 file containing the baseline data")
    parser.add_argument("--targetdir",
        required=True,
        help="Directory to store the different .npy files",
    )

    parser.add_argument("--method",
        choices=list(METHODS.keys()),
        default="clean",
        help="method used for inpainting",
    )

    args = Args(**vars(parser.parse_args()))
    polarization = 0 # TODO: Add CLI argument for this
    
    dir_root = Path(args.targetdir)
    dir_root.mkdir(parents=True, exist_ok=True)

    dir_dark = dir_root.joinpath("dark")
    dir_dark.mkdir(exist_ok=True)

    dir_eta = dir_root.joinpath("eta")
    dir_eta.mkdir(exist_ok=True)

    dir_target = dir_root.joinpath("target")
    dir_target.mkdir(exist_ok=True)


    stats = []
    with h5py.File(args.dataset, "r+") as file:
        for key in tqdm(file.keys(), total=len(file)):
            # Uncorrupted data from the telescope
            target = file[f"{key}/CORRECTED_DATA"][:, :, polarization]
            target = np.abs(target)

            # Corrupted data from the telescope
            dark = file[f"{key}/CORRUPTED_DATA"][:, :, polarization]
            dark = np.abs(dark)
            
            # In-painted result
            eta = METHODS[args.method](file, key)

            # Standardize the result according to the unflagged visibilities
            flags = file[f"{key}/FLAG"][:, :, polarization]
            mean, std = dark[~flags].mean(), dark[~flags].std()
            dark = (dark - mean) / std
            eta = (eta - mean) / std
            target = (target - mean) / std

            # Expand dimensions to be in form WxHx1
            dark = np.expand_dims(dark, axis=2)
            eta = np.expand_dims(eta, axis=2)

            # Finally, write to file
            np.save(dir_eta.joinpath(f"{key}.npy"), eta, allow_pickle=False)
            np.save(dir_dark.joinpath(f"{key}.npy"), dark, allow_pickle=False)
            np.save(dir_target.joinpath(f"{key}.npy"), target, allow_pickle=False)
            stats.append({
                "baseline": key,
                "mean": mean,
                "stdev": std,
            })

    df = pd.DataFrame(stats)
    df.set_index("baseline", inplace=True)
    df.to_csv(dir_root.joinpath("summary.csv"))


