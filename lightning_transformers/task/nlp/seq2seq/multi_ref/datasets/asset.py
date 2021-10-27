from dataclasses import dataclass
from typing import Union
from datasets import Dataset, DatasetDict
from lightning_transformers.task.nlp.seq2seq.multi_ref import MultiRefDataModule
from lightning_transformers.task.nlp.seq2seq.multi_ref.config import MultiRefConfig
from lightning_transformers.task.nlp.seq2seq.multi_ref.metric import SimplificationMetric
from lightning_transformers.task.nlp.seq2seq.multi_ref.model import MultiRefTransformer


class AssetDataModule(MultiRefDataModule):

    def split_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        if self.cfg.train_val_split is not None:
            split = dataset["validation"].train_test_split(self.cfg.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        dataset = self._select_samples(dataset)
        return dataset


@dataclass
class AssetConfig(MultiRefConfig):
    n_gram: int = 4
    smooth: bool = False


class ASSETTransformer(MultiRefTransformer):
    def __init__(
        self,
        *args,
        downstream_model_type: str = "transformers.AutoModelForSeq2SeqLM",
        cfg: AssetConfig = AssetConfig(),
        **kwargs
    ) -> None:
        super().__init__(*args, downstream_model_type=downstream_model_type, cfg=cfg, **kwargs)
        self.cfg = cfg
        self.metric = None

    def configure_metrics(self, stage: str):
        self.metric = SimplificationMetric(self.cfg.n_gram, self.cfg.smooth)
