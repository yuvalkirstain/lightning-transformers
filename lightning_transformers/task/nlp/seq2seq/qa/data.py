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

from typing import Tuple, Optional
from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataModule
from functools import partial
from datasets import Dataset


class QADataModule(Seq2SeqDataModule):
    """Defines the ``LightningDataModule`` for Summarization Datasets."""
    def preprocess_targets(self, example, idx):
        src_text_column_name, tgt_texts_column_name = self.source_targets_column_names
        src_text_column_name, tgt_text_column_name = self.source_target_column_names
        example[tgt_text_column_name] = example[tgt_texts_column_name][0]
        example[tgt_texts_column_name] = example[tgt_texts_column_name]
        example[self.idx_column_name] = idx
        return example

    def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
        dataset = dataset.map(self.preprocess_targets, with_indices=True)

        src_text_column_name, tgt_text_column_name = self.source_target_column_names

        convert_to_train_features = partial(
            self.convert_to_features,
            tokenizer=self.tokenizer,
            padding=self.cfg.padding,
            max_source_length=self.cfg.max_source_length,
            max_target_length=self.cfg.max_target_length,
            src_text_column_name=src_text_column_name,
            tgt_text_column_name=tgt_text_column_name,
        )

        dataset = dataset.map(
            convert_to_train_features,
            batched=True,
            num_proc=self.cfg.preprocessing_num_workers,
            load_from_cache_file=self.cfg.load_from_cache_file,
        )

        cols_to_keep = ["input_ids", "attention_mask", "labels"]
        dataset["train"].set_format(columns=cols_to_keep)
        cols_to_keep = ["input_ids", "attention_mask", "labels", self.idx_column_name]
        dataset["validation"].set_format(columns=cols_to_keep)

        return dataset

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return "question", "first_answer"

    @property
    def idx_column_name(self) -> str:
        return "example_id"

    @property
    def source_targets_column_names(self) -> Tuple[str, str]:
        return "question", "answer"
