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
from typing import List, Any

import torch
from datasets import load_metric
from lightning_transformers.core.nlp.seq2seq import Seq2SeqTransformer
from lightning_transformers.task.nlp.seq2seq.multi_ref.config import MultiRefConfig
from lightning_transformers.task.nlp.seq2seq.multi_ref.metric import QAMetric


class MultiRefTransformer(Seq2SeqTransformer):
    """Defines ``LightningModule`` for the Summarization Task.

    Args:
        *args: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
        downstream_model_type: Downstream HuggingFace AutoModel to load.
            (default ``transformers.AutoModelForSeq2SeqLM``)
        **kwargs: :class:`lightning_transformers.core.nlp.seq2seq.Seq2SeqTransformer` arguments.
    """

    def __init__(
        self,
        *args,
        downstream_model_type: str = "transformers.AutoModelForSeq2SeqLM",
        cfg: MultiRefConfig = MultiRefConfig(),
        **kwargs
    ) -> None:
        super().__init__(downstream_model_type, *args, cfg=cfg, **kwargs)
        self.metric = None

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        example_ids = batch.pop(self.trainer.datamodule.idx_column_name)
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        if self.cfg.compute_generate_metrics:
            batch[self.trainer.datamodule.idx_column_name] = example_ids
            self.compute_generate_metrics(batch, prefix)
        return loss

    @staticmethod
    def get_split_name_by_prefix(prefix):
        if prefix == "val":
            return "validation"
        elif prefix == "test":
            return "test"
        raise ValueError

    def compute_generate_metrics(self, batch, prefix):
        example_ids = batch.pop(self.trainer.datamodule.idx_column_name)
        split_name = self.get_split_name_by_prefix(prefix)
        examples = self.trainer.datamodule.ds[split_name].select(example_ids)
        tgt_lns = examples[self.trainer.datamodule.references_column_name]
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        result = self.metric.update(predictions=pred_lns, references=tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)

    def configure_metrics(self, stage: str):
        raise NotImplementedError

    @property
    def hf_pipeline_task(self) -> str:
        return "summarization"
