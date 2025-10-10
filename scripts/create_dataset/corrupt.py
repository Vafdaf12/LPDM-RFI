from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class CorruptConfig:
    brightness_range: tuple[float, float]
    count_range: tuple[int, int]

@dataclass
class Config:
    seed: int
    lines: CorruptConfig
    dashes: CorruptConfig
    blocks: CorruptConfig

def corrupt_images(images: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Corrupts images with randomly generated RFI, returning the result and corresponding RFI flags"""
    corrupted_images = images.copy()

    img_height, img_width = images.shape[1:3]

    np.random.seed(config.seed)
    for i in range(images.shape[0]):
        # Add solid lines of corruption
        for _ in range(np.random.randint(*config.lines.count_range)):
            brightness_factor = np.random.uniform(*config.lines.brightness_range)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 4)

            if orientation == 'horizontal':
                x = np.random.randint(0, img_width - line_width + 1)
                corrupted_images[i, x:x+line_width, :] += brightness_factor
            else:
                y = np.random.randint(0, img_height - line_width + 1)
                corrupted_images[i, :, y:y+line_width] += brightness_factor

        # Add dashed lines of corruption
        for _ in range(np.random.randint(*config.dashes.count_range)):
            brightness_factor = np.random.uniform(*config.dashes.brightness_range)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 2)
            dash_length = np.random.randint(1, img_width/2 + 1)
            gap_length = np.random.randint(1, img_width/4 + 1)
            if orientation == "horizontal":
                x = np.random.randint(0, img_height - line_width + 1)
                y = 0
                while y < img_width:
                    corrupted_images[i, x:x+line_width, y:y+dash_length] += brightness_factor
                    y += dash_length + gap_length
            else:
                x = 0
                y = np.random.randint(0, img_width - line_width + 1)
                while x < img_height:
                    corrupted_images[i, x:x+dash_length, y:y+line_width] += brightness_factor
                    x += dash_length + gap_length

        # Add solid rectangles of corruption
        for _ in range(np.random.randint(*config.blocks.count_range)):
            brightness_factor = np.random.uniform(*config.blocks.brightness_range)
            # TODO: Here is a magic constant
            block_size = np.random.randint(2, 5) * (512 // 64)
            x = np.random.randint(0, img_width - block_size + 1)
            y = np.random.randint(0, img_height - block_size + 1)
            corrupted_images[i, x:x+block_size, y:y+block_size] += brightness_factor


    corruption_mask = corrupted_images - images

    # Identify the pixels to corrupt further with Gaussian noise
    to_corrupt = corruption_mask > 0

    # Generate Gaussian noise
    noise = np.random.normal(0, 0.1, corrupted_images.shape)

    # Apply the Gaussian noise only to the pixels identified by the corruption mask
    corrupted_images[to_corrupt] += noise[to_corrupt]
    return corrupted_images, to_corrupt


