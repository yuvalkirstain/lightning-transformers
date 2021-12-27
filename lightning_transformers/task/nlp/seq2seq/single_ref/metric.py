import re
import string
from collections import Counter
from typing import List

import torch
from torchmetrics import Metric, BLEUScore


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def mean(scores):
    scores = torch.tensor(scores, dtype=torch.float)
    return torch.mean(scores)


class QAMetric(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state("exact", [])
        self.add_state("f1", [])

    def update(self, predictions: List[str], references: List[str]):
        exact = [exact_match_score(prediction, reference)
                 for prediction, reference in zip(predictions, references)]
        f1 = [f1_score(prediction, reference)
              for prediction, reference in zip(predictions, references)]
        self.exact += torch.tensor(exact)
        self.f1 += torch.tensor(f1)
        return {"exact": mean(exact), "f1": mean(f1)}

    def compute(self):
        return {"exact": mean(self.exact), "f1": mean(self.f1)}


class SimplificationMetric(Metric):
    def __init__(self, n_gram: int = 4, smooth: bool = True):
        super().__init__()
        self.bleu = BLEUScore(n_gram, smooth)

    def update(self, predictions: List[str], references: List[List[str]]):
        tgt_tokens = tuple([tuple(tuple(reference.split()) for example_references in references for reference in example_references)])
        pred_tokens = tuple(tuple(prediction.split()) for prediction in predictions)
        self.bleu.update(tgt_tokens, pred_tokens)
        result = {"bleu": self.bleu(tgt_tokens, pred_tokens).item()}
        return result

    def compute(self):
        results = self.bleu.compute()
        return results
