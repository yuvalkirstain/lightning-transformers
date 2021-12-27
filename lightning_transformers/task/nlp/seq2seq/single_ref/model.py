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


class SingleRefTransformer(Seq2SeqTransformer):
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
        outputs = self.model(**batch)
        loss, logits = outputs[:2]
        if self.cfg.compute_generate_metrics:
            self.compute_generate_metrics(batch, prefix)
        return loss

    def compute_generate_metrics(self, batch, prefix):
        pred_lns = self.generate(batch["input_ids"], batch["attention_mask"])
        tgt_lns = self.tokenize_labels(batch["labels"])
        result = self.metric.update(predictions=pred_lns, references=tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)

    def configure_metrics(self, stage: str):
        raise NotImplementedError

    @property
    def hf_pipeline_task(self) -> str:
        return "summarization"
