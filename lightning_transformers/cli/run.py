# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from functools import partial
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, seed_everything
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only

from lightning_transformers.core import TaskTransformer, TransformerDataModule
from lightning_transformers.core.config import TaskConfig, TrainerConfig, TransformerDataConfig, print_config
from lightning_transformers.core.instantiator import HydraInstantiator, Instantiator
from lightning_transformers.core.nlp.config import HFTokenizerConfig
from lightning_transformers.core.utils import set_ignore_warnings


def predict_by_batches(examples, model, source_col, generation_kwargs):
    inputs = examples[source_col]
    y = model.hf_predict(inputs, **generation_kwargs)
    return {model.hf_pipeline.return_name: [mem[f"{model.hf_pipeline.return_name}_text"] for mem in y]}


def predict(model, batch_size, split, source_col, generation_kwargs, out_path, dataset):
    data = dataset[split]
    fixed_predict_by_batches = partial(predict_by_batches,
                                       model=model,
                                       source_col=source_col,
                                       generation_kwargs=generation_kwargs)
    data = data.map(fixed_predict_by_batches, batch_size=batch_size, batched=True)
    data.to_json(out_path)


def run(
    instantiator: Instantiator,
    ignore_warnings: bool = True,
    run_fit: bool = True,
    run_validation_after_fit: bool = True,
    run_test_after_fit: bool = True,
    run_predict_after_fit: bool = False,
    dataset: TransformerDataConfig = TransformerDataConfig(),
    task: TaskConfig = TaskConfig(),
    trainer: TrainerConfig = TrainerConfig(),
    tokenizer: Optional[HFTokenizerConfig] = None,
    logger: Optional[Any] = None,
    seed: int = None,
    generation_cfg: DictConfig = None
) -> None:
    if seed:
        seed_everything(seed)

    if ignore_warnings:
        set_ignore_warnings()

    data_module_kwargs = {}
    if tokenizer is not None:
        data_module_kwargs["tokenizer"] = tokenizer

    data_module: TransformerDataModule = instantiator.data_module(dataset, **data_module_kwargs)
    if data_module is None:
        raise ValueError("No dataset found. Hydra hint: did you set `dataset=...`?")
    if not isinstance(data_module, LightningDataModule):
        raise ValueError(
            "The instantiator did not return a DataModule instance." " Hydra hint: is `dataset._target_` defined?`"
        )
    data_module.setup("fit")

    model: TaskTransformer = instantiator.model(task,
                                                tokenizer=tokenizer,
                                                model_data_kwargs=getattr(data_module, "model_data_kwargs", None),
                                                pipeline_kwargs=generation_cfg.get("pipeline_kwargs"))
    trainer = instantiator.trainer(
        trainer,
        logger=logger,
    )

    if run_fit:
        trainer.fit(model, datamodule=data_module)

    if run_validation_after_fit:
        results = trainer.validate(model, datamodule=data_module)
        print("Val Results!!!")
        print(results)

    if run_test_after_fit:
        results = trainer.test(model, datamodule=data_module)
        print("Results!!!")
        print(results)

    if run_predict_after_fit:
        # TODO this is ugly, make it nicer
        data_module: TransformerDataModule = instantiator.data_module(dataset, **data_module_kwargs)
        dataset = data_module.load_dataset()
        dataset = data_module.split_dataset(dataset)
        predict(model,
                generation_cfg.get("batch_size"),
                generation_cfg.get("split"),
                generation_cfg.get("source_col"),
                generation_cfg.get("generation_kwargs"),
                generation_cfg.get("out_path"),
                dataset)


def debugger(cfg):
    if cfg.debug.activate:
        import pydevd_pycharm
        pydevd_pycharm.settrace(cfg.debug.ip, port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)


@rank_zero_only
def log_config(cfg, logger):
    # TODO this currently only works with wandb logger...
    cfg = OmegaConf.to_object(cfg)
    cfg["working_dir"] = os.getcwd()
    logger.experiment.config.update(cfg)


def main(cfg: DictConfig) -> None:
    instantiator = HydraInstantiator()
    debugger(cfg)
    logger = instantiator.logger(cfg)
    log_config(cfg, logger)
    print_config(cfg)
    run(
        instantiator,
        ignore_warnings=cfg.get("ignore_warnings"),
        run_fit=cfg.get("training").get("run_fit"),
        run_validation_after_fit=cfg.get("training").get("run_validation_after_fit"),
        run_test_after_fit=cfg.get("training").get("run_test_after_fit"),
        run_predict_after_fit=cfg.get("training").get("run_predict_after_fit"),
        dataset=cfg.get("dataset"),
        tokenizer=cfg.get("tokenizer"),
        task=cfg.get("task"),
        trainer=cfg.get("trainer"),
        logger=logger,
        seed=cfg.get("seed"),
        generation_cfg=cfg.get("generate")
    )


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
