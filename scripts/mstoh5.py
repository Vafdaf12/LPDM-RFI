import pandas as pd
import numpy as np
import h5py
from casacore.tables import table
from dataclasses import dataclass
from typing import Tuple, List

# Utility imports for the CLI experience
import os
from tqdm import tqdm

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="HDF5 Conversion",
        description="Extracts per-baseline data from a measurement sets into an HDF5 file for faster data access",
    )
    parser.add_argument("baseline", help="CSV file containing baseline information")
    parser.add_argument("ms", help="Measurement set to extract from")
    parser.add_argument("-o", "--output",
        required=True,
        help="The destination .h5 file",
    )
    parser.add_argument("-c",
        required=True,
        nargs="+",
        help="Columns to extract from the measurement set",
        metavar="COL"
    )

    args = parser.parse_args()

    ms_path = args.ms
    out_path = args.output
    cols = args.c

    if os.path.exists(out_path):
        print(f"Output file '{out_path}' already exists. Specify a different path.")
        exit(1)

    # Open the relevant files used for extraction
    ms = table(ms_path, readonly=True)
    baselines = pd.read_csv(args.baseline)

    # Extract all the neccesary data from the dataset
    with h5py.File(out_path, "w") as h5f:
        for col in cols:
            print(f"Extracting {col} column for {len(baselines)} baselines")
            progress = tqdm(baselines.iterrows(), total=len(baselines))
            for _, b in progress:
                a1, a2 = b.a1, b.a2

                key = f"baseline_{a1}_{a2}"
                progress.set_description(key)

                group = h5f.require_group(key)
                data = ms.getcol(col, startrow=b.start, nrow=b["count"], rowincr=b.stride)
                group.create_dataset(col, data=data, compression="lzf")
