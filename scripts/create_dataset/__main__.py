# CLI Experience
from argparse import ArgumentParser
from omegaconf import OmegaConf
from dataclasses import dataclass
from tqdm import tqdm

# Data manipulation
from casacore.tables import table
import numpy as np
import corrupt
from sklearn.model_selection import train_test_split


@dataclass
class Config:
    ms_path: str
    prefix: str
    column: str
    polarization: int
    test_size: float
    corruption: corrupt.Config

def main():
    parser = ArgumentParser(description="Create training data by corrupting a measurement set with artifical RFI.")
    parser.add_argument("config",
        type=str,
        help="YAML file containing dataset configuration",
    )
    args = parser.parse_args()

    print(f"Reading configuration from {args.config}")
    config: Config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, OmegaConf.load(args.config)) #type: ignore
    config = OmegaConf.to_object(config) #type: ignore

    print(f"Reading columns: ANTENNA1, ANTENNA2, {config.column}")
    with table(config.ms_path, readonly=True) as ms:
        antenna1, antenna2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")
        data = ms.getcol(config.column)[:, :, config.polarization]

    baselines = sorted(set(zip(antenna1, antenna2)))
    baseline_data = []

    print(f"Extracting {config.column} for {len(baselines)} baseline(s)")
    progress = tqdm(baselines, total=len(baselines))
    for a1, a2 in progress:
        progress.set_description(f"baseline {a1},{a2}")

        mask = (antenna1 == a1) & (antenna2 == a2)
        baseline_data.append(np.abs(data[mask]))

    baseline_data = np.array(baseline_data)


    print(f"Corrupting baselines")
    corrupted, flags = corrupt.corrupt_images(baseline_data, config.corruption)

    print(f"Splitting into train and test data")
    corrupt_train, corrupt_test, clean_train, clean_test, flags_train, flags_test \
        = train_test_split(corrupted, baseline_data, flags, test_size=config.test_size)

    print(f"- Train set size: {corrupt_train.shape[0]}") #type: ignore
    print(f"- Test set size: {corrupt_test.shape[0]}") #type: ignore

    print(f"Outputting datasets to disk")
    np.save(f"{config.prefix}-train-input.npy", corrupt_train, allow_pickle=False)
    np.save(f"{config.prefix}-train-target.npy", clean_train, allow_pickle=False)
    np.save(f"{config.prefix}-train-flags.npy", flags_train, allow_pickle=False)

    np.save(f"{config.prefix}-test-input.npy", corrupt_test, allow_pickle=False)
    np.save(f"{config.prefix}-test-target.npy", clean_test, allow_pickle=False)
    np.save(f"{config.prefix}-test-flags.npy", flags_test, allow_pickle=False)


if __name__ == "__main__":
    main()
