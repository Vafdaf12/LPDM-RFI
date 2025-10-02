from argparse import ArgumentParser
import numpy as np

def main():
    METHODS = {
        "mean": inpaint_mean,
        "noise": inpaint_noise,
        "zero": inpaint_zero,
    }

    parser = ArgumentParser()
    parser.add_argument("name", type=str, help="name prefix of the dataset")
    parser.add_argument("--method",
        choices=list(METHODS.keys()),
        help="method used for inpainting",
    )
    args = parser.parse_args()

    print("Loading datasets:")
    print(f"- {args.name}-input.npy")
    print(f"- {args.name}-flags.npy")
    np.random.seed(1)
    inputs = np.load(f"{args.name}-input.npy")
    flags = np.load(f"{args.name}-flags.npy")


    print(f"Inpainting using '{args.method}' method...")
    eta = METHODS[args.method](inputs, flags)

    print(f"Saving results to '{args.name}-inpaint-{args.method}.npy'")
    np.save(f"{args.name}-inpaint-{args.method}.npy", eta, allow_pickle=False)

def inpaint_mean(inputs: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """In-paints an RFI-corrupted baseline using the mean value of the unflagged data"""
    eta = inputs.copy()
    for i in range(inputs.shape[0]):
        mask = flags[i]
        eta[i][mask] = eta[i][~mask].mean()

    return eta

def inpaint_zero(inputs: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """In-paints an RFI-corrupted baseline by zeroing out flagged regions"""
    eta = inputs.copy()
    for i in range(inputs.shape[0]):
        mask = flags[i]
        eta[i][mask] = 0

    return eta

def inpaint_noise(inputs: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """In-paints an RFI-corrupted baseline with randomly generated gaussian noise"""
    eta = inputs.copy()
    for i in range(inputs.shape[0]):
        mask = flags[i]
        mean, stdev = eta[i][~mask].mean(), eta[i][~mask].std()
        noise = np.random.normal(mean, stdev, size=eta[i].shape)
        eta[i][mask] = noise[mask]

    return eta


if __name__ == "__main__":
    main()
