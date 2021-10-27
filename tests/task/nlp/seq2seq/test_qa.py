import sys
from unittest.mock import MagicMock

import pytest

from lightning_transformers.core.nlp import HFBackboneConfig
from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataConfig
from lightning_transformers.task.nlp.seq2seq.qa import MultiRefDataModule, MultiRefTransformer
from lightning_transformers.task.nlp.seq2seq.qa.config import QAConfig


@pytest.mark.skipif(sys.platform == "win32", reason="Currently Windows is not supported")
def test_smoke_train_e2e(script_runner):
    script_runner.hf_train(task="seq2seq/qa", dataset="nq_open", model="patrickvonplaten/t5-tiny-random")


def test_smoke_predict_e2e(script_runner):
    y = script_runner.hf_predict(
        [
            '+x="The results found significant improvements over all tasks evaluated"',
            "+predict_kwargs={min_length: 2, max_length: 12}",
        ],
        task="seq2seq/qa",
        model="patrickvonplaten/t5-tiny-random",
    )
    assert len(y) == 1
    output = y[0]["summary_text"]
    assert 2 <= len(output.split()) <= 12


def test_model_has_correct_cfg():
    model = MultiRefTransformer(HFBackboneConfig(pretrained_model_name_or_path="t5-base"))
    assert model.hparams.downstream_model_type == "transformers.AutoModelForSeq2SeqLM"
    assert type(model.cfg) is QAConfig


def test_datamodule_has_correct_cfg():
    tokenizer = MagicMock()
    dm = MultiRefDataModule(tokenizer)
    assert type(dm.cfg) is Seq2SeqDataConfig
    assert dm.tokenizer is tokenizer
