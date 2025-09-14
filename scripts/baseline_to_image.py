import h5py

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
    outfile = args.output
    polarization = 0 # TODO: Add CLI argument for this

    with h5py.File(infile, "r+") as file:
        # Data coming from the telescope
        dark = file[f"{baselines}/CORRUPTED_DATA"][:, :, polarization]
        dark = np.abs(dark)

        # CLEAN model predicted from corrupted data
        model_data = file[f"{baselines}/MODEL_DATA"][:, :, polarization]
        model_data = np.abs(model_data)

        # In-painted result
        eta = np.copy(dark)
        flags = file[f"{baselines}/FLAG"][:, :, polarization]
        eta[flags] =  model_data[flags]


        mean, std = dark[~flags].mean(), dark[~flags].std()
        dark = (dark - mean) / std
        eta = (eta - mean) / std

        print(dark.shape, eta.shape)


