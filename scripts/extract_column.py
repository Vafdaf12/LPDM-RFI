import pandas as pd
import numpy as np
import h5py
from casacore.tables import table
from pathlib import Path
from typing import Tuple, List

# Utility imports for the CLI experience
import os
from tqdm import tqdm

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="HDF5 Conversion",
        description="Extract per-baseline data from a measurement set into a collection of .npy files",
    )
    parser.add_argument("baseline", help="CSV file containing baseline information from extract_baselines.py")
    parser.add_argument("ms", help="measurement set to extract from")
    parser.add_argument("-o", "--output",
        type=Path,
        required=True,
        help="directory where extracted data will be stored",
    )
    parser.add_argument("-c", "--column",
        required=True,
        help="column to extract",
        metavar="COL"
    )

    args = parser.parse_args()

    baseline = args.baseline
    ms_path = args.ms
    out_path = args.output
    col = args.column


    # Set up output directory
    out_path.mkdir(parents=True, exist_ok=False)

    # Open the relevant files used for extraction
    baselines = pd.read_csv(baseline)

    # Extract all the neccesary data from the dataset
    with table(ms_path, readonly=True) as ms:
        print(f"Reading {col} column for {len(baselines)} baselines")
        all_data = ms.getcol(col)


        print(f"Extracting {len(baselines)} baselines")
        progress = tqdm(baselines.iterrows(), total=len(baselines))
        for _, b in progress:
            a1, a2 = b.a1, b.a2
            key=f"baseline_{a1}_{a2}.npy"
            progress.set_description(key)

            data = all_data[b.start::b.stride][:b["count"]]
            np.save(out_path.joinpath(key), data, allow_pickle=False)