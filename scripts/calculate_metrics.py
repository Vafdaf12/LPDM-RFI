# Must run from within the scripts directory
from skimage.metrics import peak_signal_noise_ratio
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

def evaluate(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "psnr": peak_signal_noise_ratio(pred, target, data_range=target.max()),
        "mae": np.mean(np.absolute(pred - target))
    }

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('pred',
        type=Path,
        help="dataset containing predictions from the model"
    )
    parser.add_argument('-t', '--target',
        type=Path,
        required=True,
        help="dataset containing the target outputs of the model"
    )
    parser.add_argument('-o', '--output',
        type=Path,
        required=True,
        help="location to output the calculated metrics"
    )
    args = parser.parse_args()

    pred, target = np.load(args.pred), np.load(args.target)
    assert len(pred.shape) == len(target.shape), f"Shape of datasets do not match: {pred.shape} != {target.shape}"

    print("Calculating metrics...")
    results = []
    for pred, target in zip(pred, target):
        metrics = evaluate(pred, target)
        results.append(metrics)

    df = pd.DataFrame(results)
    means = df.mean()
    stdevs = df.std()
    for i, (m, s) in enumerate(zip(means, stdevs)):
        print(f"{means.index[i]}:\t {m} ± {s}")

    df.to_csv(args.output)

# print(f'Scanning configs...')
# results_csv = '../test/results.csv'

# fr = FullReferenceMeasure()
# nr = NoReferenceMeasure()

# # for config_path in glob.glob('../configs/metrics/**/*.yaml', recursive=True):
# config_path = args.base_path

# if not args.no_skip and os.path.isfile(results_csv):
#     current_results = pd.read_csv(results_csv, index_col='file')
#     if os.path.basename(config_path) in current_results.index:
#         print(f'Skipping: {os.path.basename(config_path)}')
#         exit()

# print(f'Calculating metrics: {os.path.basename(config_path)}')
# config = OmegaConf.load(config_path)
# pred_paths = glob.glob(config.pred_path)

# results_fr = {}
# results_nr= {}

# if 'target_path' in config:
#     target_paths = glob.glob(config.target_path)
#     assert len(pred_paths) == len(target_paths), f"Number of images in folders to not match {len(pred_paths)} != {len(target_paths)}"
#     for p, y in zip(pred_paths, target_paths):
#         assert os.path.splitext(os.path.basename(p))[0] == os.path.splitext(os.path.basename(y))[0] # Ignore extension

#     results_fr = fr.get_empty_metrics()
#     for p_path, y_path  in tqdm(zip(pred_paths, target_paths)):
#         # p, y = imread(p_path, as_gray=True), imread(y_path, as_gray=True)
#         p, y = imread(p_path), imread(y_path)
#         metrics = fr.measure(p,y)
#         for k,v in metrics.items():
#             results_fr[k].append(v)

# if 'target_path' not in config:
#     results_nr = nr.get_empty_metrics()
#     # Not calculating NR metrics for paired data
#     for p_path in tqdm(pred_paths):
#         metrics = nr.measure(p_path) # Uses paths and not np arrays
#         for k,v in metrics.items():
#             results_nr[k].append(v)

# all_metrics = {**results_fr, **results_nr}
# results = {k: [np.mean(v)] for k,v in all_metrics.items()}
# df = pd.DataFrame.from_dict(results, orient='columns')

# if 'model' in config:
#     df.insert(0, "model", config.model)
# df['dataset'] = config.dataset
# df['description'] = config.description
# df['file'] = os.path.basename(config_path)
# df = df.set_index('file')

# head, _ = os.path.split(config.pred_path)
# denosing_configs = glob.glob(os.path.join(head, '*.yaml')) # denoiser writes its config to the pred dir
# if denosing_configs:
#     assert len(denosing_configs) == 1, "Found too many denoising configs"
#     denoising_config = OmegaConf.load(denosing_configs[0])
#     df['t'] = denoising_config.t
#     if 's' in denoising_config:
#         df['s'] = denoising_config.s
#     df['ddpm_name'] = denoising_config.ddpm_name

# if not os.path.isfile(results_csv):
#     df.to_csv(results_csv, header='column_names')
# else:
#     current = pd.read_csv(results_csv, index_col='file')
#     current = pd.concat([current, df], join='outer')
#     # current = pd.concat([current, df], how='outer', on='file')
#     current.to_csv(results_csv)

# print("Metric calculations complete!")
