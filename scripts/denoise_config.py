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
    print(f"Loading model from {ckpt}")
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

def load_model_given_name(name, device = torch.device('cpu')):
    config_path = os.path.join('../checkpoints',  f'{name}.yaml')
    config = OmegaConf.load(config_path)
    if 'first_stage_config' in config.model.params and not config.model.params.first_stage_config.params.ckpt_path.startswith('..') :
        config.model.params.first_stage_config.params.ckpt_path = os.path.join('..', config.model.params.first_stage_config.params.ckpt_path)
    model = load_model_from_config(config, f'{config_path[:-5]}.ckpt') # Assume the config is the same name
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
    ddpm_name: str
    pred_path: Path
    cond_path: Path
    write_path: Path
    description: str
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
    output_path = config.write_path / f'phi{config.phi}_s{config.s}'
    assert args.overwrite or not output_path.exists(), f"Output file {output_path} already exists, but --overwrite was not set."
    config.write_path.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    preds = np.load(config.pred_path)
    conds = np.load(config.cond_path)
    assert len(preds.shape) == len(conds.shape), f"Shape of datasets do not match: {preds.shape} != {conds.shape}"

    print(f"Loading model with name {config.ddpm_name}...")
    device = torch.device(args.device)
    ddpm = load_model_given_name(config.ddpm_name).to(device)

    print(f'Denoising using phi={config.phi}, s={config.s}...')
    outputs = []
    with torch.no_grad():
        for p, c in tqdm(zip(preds, conds), total=preds.shape[0]):
            # Transform into WxHx1, then 1xHxW
            p, c = np.expand_dims(p, axis=2), np.expand_dims(c, axis=2)
            p, c = np.swapaxes(p, 0, 2), np.swapaxes(c, 0, 2)

            # Transform into batch of size 1 (sad)
            p, c = numpy_to_tensor(p).unsqueeze(0), numpy_to_tensor(c).unsqueeze(0)
            p, c = p.to(device), c.to(device)

            t = torch.tensor([config.phi], dtype=torch.long, device=device)
            try:
                noise_pred = ddpm.model(torch.cat([p, c], dim=1), t).detach()
                x0 = ddpm.predict_start_from_noise(p, torch.tensor([config.s], device=device).long(), noise_pred).detach()
            except:
                h, w = p.shape[-2], p.shape[-1]
                noise_pred = ddpm.model(torch.cat([pad_to_multiple(p), pad_to_multiple(c)], dim=1), t).detach()
                x0 = ddpm.predict_start_from_noise(pad_to_multiple(p), torch.tensor([config.s], device=device).long(), noise_pred).detach()
                x0 = x0[..., :h, :w]


            output = tensor_to_npy(x0)
            print(output.shape)
            raise Exception("Testing shape")

    print("Denoising Complete!")

if __name__ == "__main__":
    main()
