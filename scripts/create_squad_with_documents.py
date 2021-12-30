import logging
from multiprocessing import Manager
import wikipedia
import datasets
from fire import Fire
from tqdm import tqdm

WINDOW_LEN = 20

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# manager = Manager()
# title2content = manager.dict()
title2content = {}


def main(dataset_name: str = "yuvalkirstain/squad_full_doc",
         local_path="../data/squad_with_document",
         num_proc=50,
         start_examples=0,
         end_examples=100000):
    squad = datasets.load_dataset("squad")
    bad_titles = set()
    for split in ["train", "validation"]:
        examples = squad[split]
        examples = examples.select(range(start_examples, min(len(examples), end_examples)))

        pbar = tqdm(examples)
        for example in pbar:
            title = example["title"]
            if title not in title2content and title not in bad_titles:
                try:
                    title2content[title] = wikipedia.page(title, auto_suggest=False).content
                except Exception as e:
                    bad_titles.add(title)
                    logger.info(f"title: {title} ; Exception: {e}")
            pbar.set_description(f"{len(title2content)} titles")

        examples = examples.map(update_example, num_proc=num_proc)
        examples = examples.filter(lambda example: example["valid"])
        examples = examples.remove_columns("valid")
        logger.info(f"{len(examples)} for {split}")
        squad[split] = examples
        squad[split].to_json(f"{local_path}.json")

    squad.push_to_hub(dataset_name)


def update_example(example):
    title = example["title"]
    answers_text = example["answers"]["text"]
    answers_start = example["answers"]["answer_start"]
    context = example["context"]

    content = title2content.get(title, None)
    if not content:
        example["valid"] = False
        return example

    texts, starts = [], []

    for answer, answer_start in zip(answers_text, answers_start):

        surroundings = context[max(answer_start - WINDOW_LEN, 0):
                               min(answer_start + len(answer) + WINDOW_LEN, len(context))]

        if surroundings not in content:
            example["valid"] = False
            return example

        new_start = content.index(surroundings) + surroundings.index(answer)
        texts += [answer]
        starts += [new_start]

    example["answers"]["answer_start"], example["answers"]["text"] = starts, texts
    example["context"] = content
    example["valid"] = True

    return example


if __name__ == '__main__':
    Fire(main)
