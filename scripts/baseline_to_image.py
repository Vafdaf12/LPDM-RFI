import h5py
import numpy as np

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


METHODS = {
    "clean": inpaint_clean,
    "mean": inpaint_mean,
    "zero": inpaint_zero,
}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="Baseline to Image Conversion",
        description="In-paints a baseline from the input dataset, writing the result to a .npy file for processing",
    )
    parser.add_argument("input", help=".h5 file containing the baseline data")
    parser.add_argument("base", help="name of the baseline to extract")
    parser.add_argument("--name",
        required=True,
        help="Prefix of output .npy files",
    )
    parser.add_argument("--method",
        choices=list(METHODS.keys()),
        default="clean",
        help="method used for inpainting",
    )

    args = parser.parse_args()
    infile = args.input
    baseline = args.base
    name = args.name
    method = args.method
    polarization = 0 # TODO: Add CLI argument for this

    with h5py.File(infile, "r+") as file:
        # Data coming from the telescope
        dark = file[f"{baseline}/CORRUPTED_DATA"][:, :, polarization]
        dark = np.abs(dark)
        
        # In-painted result
        eta = METHODS[method](file, baseline)

        # Standardize the result according to the unflagged visibilities
        flags = file[f"{baseline}/FLAG"][:, :, polarization]
        mean, std = dark[~flags].mean(), dark[~flags].std()
        dark = (dark - mean) / std
        eta = (eta - mean) / std

        # Expand dimensions to be in form WxHx1
        dark = np.expand_dims(dark, axis=2)
        eta = np.expand_dims(eta, axis=2)

        # Finally, write to file
        np.save(f"{name}-dark.npy", dark, allow_pickle=False)
        np.save(f"{name}-eta.npy", eta, allow_pickle=False)


