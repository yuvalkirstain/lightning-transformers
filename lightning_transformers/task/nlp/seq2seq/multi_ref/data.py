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

from typing import Tuple, Optional, Union
from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataModule
from functools import partial
from datasets import Dataset, DatasetDict

from lightning_transformers.task.nlp.seq2seq.multi_ref.config import MultiRefDataConfig


class MultiRefDataModule(Seq2SeqDataModule):
    cfg: MultiRefDataConfig

    """Defines the ``LightningDataModule`` for Summarization Datasets."""
    def preprocess_targets(self, example, idx):
        example[self.target_column_name] = example[self.references_column_name][0]
        example[self.idx_column_name] = idx
        return example

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        dataset = dataset.map(self.preprocess_targets, with_indices=True)

        convert_to_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.cfg.padding,
            max_source_length=self.cfg.max_source_length,
            max_target_length=self.cfg.max_target_length,
            src_text_column_name=self.source_column_name,
            tgt_text_column_name=self.target_column_name,
        )

        dataset = dataset.map(
            convert_to_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        cols_to_keep = ["input_ids", "attention_mask", "labels", self.idx_column_name]
        dataset.set_format(columns=cols_to_keep)

        return dataset

    @property
    def source_column_name(self) -> str:
        return self.cfg.source_column_name

    @property
    def target_column_name(self) -> str:
        return self.cfg.target_column_name

    @property
    def references_column_name(self) -> str:
        return self.cfg.references_column_name

    @property
    def idx_column_name(self) -> str:
        return self.cfg.idx_column_name

