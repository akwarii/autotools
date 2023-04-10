from shutil import rmtree
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from config import AutoToolsConfig

cs = ConfigStore.instance()
cs.store(name="autotools_config", node=AutoToolsConfig)

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: AutoToolsConfig):
    print(cfg.calculators)
    rmtree("outputs")
    
if __name__ == "__main__":
    main()