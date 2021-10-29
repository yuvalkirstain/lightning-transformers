import logging
import os
import subprocess
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import re

log = logging.getLogger(__name__)


def add_slurm(cfg):
    return f"""#SBATCH --job-name={cfg.job_name}
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --time={cfg.time}
#SBATCH --signal=USR1@120
#SBATCH --partition="killable"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000 
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n-101,n-007,n-201
#SBATCH --gpus={cfg.gpus}"""


def add_path_to_workdir_in_not_none(path):
    if path is None:
        return None
    # TODO we need to fix this and allow running from anywhere
    working_directory = get_original_cwd()
    return os.path.join(working_directory, path)



@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    slurm_name = "slurm.sh"

    with open(slurm_name, "w") as f:
        script = f"""#!/bin/bash -x

{add_slurm(cfg) if cfg.run_on_slurm else ""}

python {add_path_to_workdir_in_not_none("run.py")} \
task={cfg.task} \
dataset={cfg.dataset} \
trainer.gpus={cfg.gpus} \
dataset.cfg.limit_train_samples={cfg.limit_train_samples} \
seed={cfg.seed} \
training.run_fit={cfg.run_fit} \
training.run_validation_after_fit={cfg.run_validation_after_fit} \
training.run_predict_after_fit={cfg.run_predict_after_fit} \
dataset.cfg.dataset_name={cfg.dataset_name} \
dataset.cfg.train_file={add_path_to_workdir_in_not_none(cfg.train_file)} \
dataset.cfg.validation_file={add_path_to_workdir_in_not_none(cfg.validation_file)} \
dataset.cfg.test_file={add_path_to_workdir_in_not_none(cfg.test_file)}
"""
        script = script.replace("None", "null")
        f.write(script)

    if cfg.dry:
        log.info(script)
        return

    process = subprocess.Popen(["sbatch" if cfg.run_on_slurm else "bash", slurm_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    log.info("output:")
    output = stdout.decode("utf-8")

    log.info(output)
    log.info("err:")
    err = stderr.decode("utf-8")
    log.info(err)

    job_id = re.sub("[^0-9]", "", output)
    if len(err) == 0:
        process = subprocess.Popen(["scontrol", "show", "job", job_id, "|", "grep", job_id], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = stdout.decode("utf-8")
        print(output)


if __name__ == '__main__':
    main()
