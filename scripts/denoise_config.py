# Must run from within the 'scripts' directory
import os, sys
import glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
import torch
import torchvision.transforms as T
import argparse
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath(os.path.join('..', 'external', 'taming-transformers')))
sys.path.append(os.path.abspath(os.path.join('..', 'external', 'clip')))
from ldm.util import instantiate_from_config

from dataclasses import dataclass
from pathlib import Path


def tensor_to_pil(t, mode='RGB'):
    if t.ndim == 4:
        assert t.shape[0] == 1
        t = t[0]
    t = torch.clamp((t + 1.0) / 2.0, min=0.0, max=1.0)
    img = 255. * rearrange(t.detach().cpu().numpy(), 'c h w -> h w c')
    return Image.fromarray(img.astype(np.uint8), mode=mode).convert('RGB')

def tensor_to_npy(t):
    arr = t.cpu().numpy()[0]
    arr = np.swapaxes(arr, 0, 2)
    return arr



def is_numpy_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False

    ext = os.path.splitext(path)[1]
    return ext in (".npy", ".npz")

def numpy_to_tensor(t):

    return torch.from_numpy(t)
    #return T.ToTensor()(t)

def pil_to_tensor_in_range(t):
    return T.ToTensor()(t) * 2.0 - 1.0

def clamp_ldm_range(t):
    return torch.clamp(t, -1.0, 1.0)

def ldm_range_to_rgb_range(t):
    t = clamp_ldm_range(t)
    return (t + 1.0) / 2.0

def get_diff_loss_prediction(diffloss_model, z_m, z_x):
    t_ = torch.full((z_x.shape[0],), 0).long()
    return diffloss_model.diffusion_model.apply_model(torch.cat([z_m, z_x],dim=1), t_, cond=None).detach()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model checkpoint from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def load_model_from_path(config_path: str, ckpt_path: str, device = torch.device('cpu')):
    print(f"Loading model configuration from {config_path}")
    config = OmegaConf.load(config_path)
    if 'first_stage_config' in config.model.params and not config.model.params.first_stage_config.params.ckpt_path.startswith('..') :
        config.model.params.first_stage_config.params.ckpt_path = os.path.join('..', config.model.params.first_stage_config.params.ckpt_path)

    model = load_model_from_config(config, ckpt_path)
    model = model.to(device)
    _ = model.eval()
    return model

def pad_to_multiple(im, mul=16):
    h, w = im.shape[2], im.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    return torch.nn.functional.pad(im, (0, padw, 0, padh), mode='reflect')


@dataclass
class Config:
    checkpoint_path: str
    config_path: str

    pred_path: str
    cond_name: str
    write_path: str
    s: int
    phi: int


def main():
    seed_everything(0)
    parser = argparse.ArgumentParser(prog = 'Denoise Config', description = 'Denoise folders of images based on a configuration file.')

    parser.add_argument('-b', '--base_path', type=str, default='../configs/test/denoise.yaml', help='a string path to the test configuration file')
    parser.add_argument('-d', '--device', type=str, help='device to move the model and tensors to. Either cuda or cpu.', default='cpu')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing images rather than skipping the denoising process.')
    args = parser.parse_args()

    print(f"Reading configuration from {args.base_path}")
    config: Config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, OmegaConf.load(args.base_path)) #type: ignore
    config = OmegaConf.to_object(config) #type: ignore

    print("Checking configuration...")
    output_path = Path(config.write_path) / f'phi{config.phi}_s{config.s}.npy'
    assert args.overwrite or not output_path.exists(), f"Output file {output_path} already exists, but --overwrite was not set."
    Path(config.write_path).mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    preds = np.load(config.pred_path)
    conds = np.load(f"{config.cond_name}-input.npy")
    flags = np.load(f"{config.cond_name}-flags.npy")
    assert len(preds.shape) == len(conds.shape), f"Shape of datasets do not match: {preds.shape} != {conds.shape}"

    print(f"Loading model...")
    device = torch.device(args.device)
    ddpm = load_model_from_path(config.config_path, config.checkpoint_path).to(device)

    print(f'Denoising using phi={config.phi}, s={config.s}...')
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(preds.shape[0]), total=preds.shape[0]):
            pred, cond, flag = preds[i], conds[i], flags[i]

            # Standardize according to the unflagged condition
            mean, stdev = np.mean(cond[~flag]), np.std(cond[~flag])
            pred = (pred - mean) / stdev
            cond = (cond - mean) / stdev

            # Transform into WxHx1, then 1xHxW
            pred, cond = np.expand_dims(pred, axis=2), np.expand_dims(cond, axis=2)
            pred, cond = np.swapaxes(pred, 0, 2), np.swapaxes(cond, 0, 2)

            # Transform into batch of size 1 (sad)
            pred, cond = numpy_to_tensor(pred).unsqueeze(0), numpy_to_tensor(cond).unsqueeze(0)
            pred, cond = pred.to(device), cond.to(device)

            t = torch.tensor([config.phi], dtype=torch.long, device=device)
            try:
                noise_pred = ddpm.model(torch.cat([pred, cond], dim=1), t).detach()
                x0 = ddpm.predict_start_from_noise(pred, torch.tensor([config.s], device=device).long(), noise_pred).detach()
            except:
                h, w = pred.shape[-2], pred.shape[-1]
                noise_pred = ddpm.model(torch.cat([pad_to_multiple(pred), pad_to_multiple(cond)], dim=1), t).detach()
                x0 = ddpm.predict_start_from_noise(pad_to_multiple(pred), torch.tensor([config.s], device=device).long(), noise_pred).detach()
                x0 = x0[..., :h, :w]


            output = tensor_to_npy(x0)[:, :, 0]
            output = mean + (output * stdev)
            outputs.append(output)

    print(f"Denoising Complete! Saving results to {output_path}")
    outputs = np.array(outputs)
    np.save(output_path, outputs, allow_pickle=False)

if __name__ == "__main__":
    main()
