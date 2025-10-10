from dataclasses import dataclass
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pathlib import Path

@dataclass
class OutputConfig:
    checkpoint_path: str
    config_path: str

    pred_path: str
    cond_name: str
    write_path: str
    s: int
    phi: int

@dataclass
class Config:
    checkpoint_path: str
    config_path: str

    pred_path: str
    cond_name: str
    write_path: str
    phis: list[int]
    ss: list[int]

    def create_output(self, phi: int, s: int):
        return OutputConfig(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            pred_path=self.pred_path,
            cond_name=self.cond_name,
            write_path=self.write_path,
            s=s,
            phi=phi,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("config",
        type=str,
        help="YAML file containing grid configuration",
    )
    parser.add_argument("--output",
        type=Path,
        required=True,
        help="Output folder of the generated config files"
    )

    args = parser.parse_args()

    print(f"Reading configuration from {args.config}")
    config: Config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, OmegaConf.load(args.config)) #type: ignore
    config = OmegaConf.to_object(config) #type: ignore


    outdir: Path = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    for s in config.ss:
        for phi in config.phis:
            outpath = outdir / f"phi{phi}_s{s}.yaml"
            print(f"Generating {outpath}")
            conf = config.create_output(phi, s)
            OmegaConf.save(conf,  outpath)


if __name__ == "__main__":
    main()
