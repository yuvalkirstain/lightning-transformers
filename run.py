"""The shell entry point `$ pl-transformers-train` is also available."""
import hydra
from omegaconf import DictConfig

from lightning_transformers.cli.run import main
from lightning_transformers.core.config import print_config


@hydra.main(config_path="./conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    print_config(cfg)
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
