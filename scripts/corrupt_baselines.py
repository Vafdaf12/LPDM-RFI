import numpy as np
import h5py

import argparse
from tqdm import tqdm

LINE_BRIGHTNESS = (0.05, 0.20)
LINE_COUNT = (0, 4)

DASH_BRIGHTNESS = (0.10, 0.50)
DASH_COUNT = (1, 10)

BLOCK_BRIGHTNESS = (0.10, 0.50)
BLOCK_COUNT = (1, 3)

FLAG_COL = "FLAG"
OUTPUT_COL = "CORRUPTED_DATA"

def corrupt_images(images: np.ndarray) -> np.ndarray:
    corrupted_images = images.copy()
    
    img_height, img_width = images.shape[1:3]

    for i in range(images.shape[0]):
        # Add solid lines of corruption
        for _ in range(np.random.randint(*LINE_COUNT)):
            brightness_factor = np.random.uniform(*LINE_BRIGHTNESS)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 4)

            if orientation == 'horizontal':
                x = np.random.randint(0, img_width - line_width + 1)
                corrupted_images[i, x:x+line_width, :, :] += brightness_factor
            else:
                y = np.random.randint(0, img_height - line_width + 1)
                corrupted_images[i, :, y:y+line_width, :] += brightness_factor

        # Add dashed lines of corruption
        for _ in range(np.random.randint(*DASH_COUNT)):
            brightness_factor = np.random.uniform(*DASH_BRIGHTNESS)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 2)
            dash_length = np.random.randint(1, img_width/2 + 1)
            gap_length = np.random.randint(1, img_width/4 + 1)
            if orientation == "horizontal":
                x = np.random.randint(0, img_height - line_width + 1)
                y = 0
                while y < img_width:
                    corrupted_images[i, x:x+line_width, y:y+dash_length, :] += brightness_factor
                    y += dash_length + gap_length
            else:
                x = 0
                y = np.random.randint(0, img_width - line_width + 1)
                while x < img_height:
                    corrupted_images[i, x:x+dash_length, y:y+line_width, :] += brightness_factor
                    x += dash_length + gap_length
        
        # Add solid rectangles of corruption
        for _ in range(np.random.randint(*BLOCK_COUNT)):
            brightness_factor = np.random.uniform(*BLOCK_BRIGHTNESS)
            block_size = np.random.randint(2, 5) * (512 // 64)
            x = np.random.randint(0, img_width - block_size + 1)
            y = np.random.randint(0, img_height - block_size + 1)
            corrupted_images[i, x:x+block_size, y:y+block_size, :] += brightness_factor

    
    corruption_mask = corrupted_images - images

    # Identify the pixels to corrupt further with Gaussian noise
    to_corrupt = corruption_mask > 0

    # Generate Gaussian noise
    noise = np.random.normal(0, 0.1, corrupted_images.shape)
        
    # Apply the Gaussian noise only to the pixels identified by the corruption mask
    corrupted_images[to_corrupt] += noise[to_corrupt]
    return corrupted_images, to_corrupt

if __name__ == "__main__":
    # These imports are only needed when executed as a CLI, hence the conditional import
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser(
        prog="Baseline RFI Corruption",
        description="Corrupts spectrograms in an HDF5 dataset with random RFI",
        epilog=f"Corruption data will be written to {FLAG_COL} and {OUTPUT_COL} dataset for each baseline."
    )


    parser.add_argument("input", help="HDF5 file to corrupt")
    parser.add_argument("colname", help="Name of the column to corrupt")

    parser.add_argument("--seed",
        type=int,
        default=1,
        help="The random seed to use for corruption",
    )

    args = parser.parse_args()
    file_input = args.input
    colname = args.colname
    seed = args.seed

    print(f"Corrupting baselines from {colname} column (Seed: {seed})")
    np.random.seed(seed)
    with h5py.File(file_input, "r+") as file:
        progress = tqdm(file.keys(), total=len(file))
        for baseline in progress:
            key = f"{baseline}/{colname}"

            progress.set_description(key)

            data = file[key][...]
            dirty, flag = corrupt_images(np.expand_dims(data, 0))
            dirty, flag = dirty[0], flag[0]

            group = file.require_group(baseline)
            group.create_dataset(FLAG_COL, data=flag, compression="lzf")
            group.create_dataset(OUTPUT_COL, data=dirty, compression="lzf")
