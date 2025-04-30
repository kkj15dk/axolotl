"""Training and evaluation"""

import hydra
import os
import numpy as np
import axolotl.run_train as run_train
import axolotl.utils as utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    ngpus = config.ngpus
    if config.load_dir is not None:
        work_dir = config.load_dir # might be a bad way to load, but it works
        assert os.path.exists(work_dir), f"Load dir {work_dir} does not exist"

        hydra_config_path = os.path.join(config.load_dir, ".hydra/hydra.yaml")
        hydra_config = OmegaConf.load(hydra_config_path).hydra

        config = utils.load_hydra_config_from_run(config.load_dir)
    else:
        hydra_config = HydraConfig.get()
        work_dir = hydra_config.run.dir if hydra_config.mode == RunMode.RUN else os.path.join(hydra_config.sweep.dir, hydra_config.sweep.subdir)
        utils.makedirs(work_dir)

    with open_dict(config):
        config.ngpus = ngpus
        config.work_dir = work_dir # doesn't seem to update the actual file

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_config = HydraConfig.get()
    if hydra_config.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_config.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_train.run_multiprocess, args=(ngpus, config, port), nprocs=ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()