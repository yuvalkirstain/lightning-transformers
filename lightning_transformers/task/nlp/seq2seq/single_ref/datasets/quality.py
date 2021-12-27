from dataclasses import dataclass
from typing import Union
from datasets import Dataset, DatasetDict

from lightning_transformers.core.nlp.seq2seq import HFSeq2SeqConfig
from lightning_transformers.task.nlp.seq2seq.single_ref.metric import QAMetric
from lightning_transformers.task.nlp.seq2seq.single_ref import SingleRefDataModule
from lightning_transformers.task.nlp.seq2seq.single_ref.model import SingleRefTransformer


class QualityDataModule(SingleRefDataModule):

    def preprocess_dataset(self, example, idx):
        return example



@dataclass
class QualityConfig(HFSeq2SeqConfig):
    n_gram: int = 4
    smooth: bool = False


class QualityTransformer(SingleRefTransformer):
    def __init__(
        self,
        *args,
        downstream_model_type: str = "transformers.AutoModelForSeq2SeqLM",
        cfg: QualityConfig = QualityConfig(),
        **kwargs
    ) -> None:
        super().__init__(*args, downstream_model_type=downstream_model_type, cfg=cfg, **kwargs)
        self.cfg = cfg
        self.metric = None

    def configure_metrics(self, stage: str):
        self.metric = QAMetric()
