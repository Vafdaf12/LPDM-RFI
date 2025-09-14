import h5py
import numpy as np

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

    args = parser.parse_args()
    infile = args.input
    baseline = args.base
    name = args.name
    polarization = 0 # TODO: Add CLI argument for this

    with h5py.File(infile, "r+") as file:
        # Data coming from the telescope
        dark = file[f"{baseline}/CORRUPTED_DATA"][:, :, polarization]
        dark = np.abs(dark)

        # CLEAN model predicted from corrupted data
        model_data = file[f"{baseline}/MODEL_DATA"][:, :, polarization]
        model_data = np.abs(model_data)

        # In-painted result
        eta = np.copy(dark)
        flags = file[f"{baseline}/FLAG"][:, :, polarization]
        eta[flags] =  model_data[flags]


        # Standardize the result according to the telescope response
        mean, std = dark[~flags].mean(), dark[~flags].std()
        dark = (dark - mean) / std
        eta = (eta - mean) / std

        # Expand dimensions to be in form WxHx1
        dark = np.expand_dims(dark, axis=2)
        eta = np.expand_dims(eta, axis=2)

        # Finally, write to file
        np.save(f"{name}-dark.npy", dark, allow_pickle=False)
        np.save(f"{name}-eta.npy", eta, allow_pickle=False)


