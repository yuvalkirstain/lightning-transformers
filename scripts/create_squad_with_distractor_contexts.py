import json
from datasets import load_dataset
from random import seed, sample, randrange
import os
from tqdm import tqdm
import logging
from fire import Fire

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def write_squad(directory):
    squad = load_dataset("squad")
    contexts = squad["train"]["context"]
    seed(42)
    os.makedirs(directory, exist_ok=True)
    for i in tqdm([0, 1, 2, 4, 8, 16, 32, 64]):
        length_dir = os.path.join(directory, str(i))
        if os.path.exists(length_dir):
            logger.info(f"skipping {i} - directory exists")
            continue
        os.makedirs(length_dir, exist_ok=True)
        for split in ["train", "validation"]:
            with open(os.path.join(length_dir, f"{split}.json"), "w") as f:
                for example in squad[split]:
                    context = example["context"]

                    sampled_contexts = sample(contexts, i)
                    idx = randrange(len(sampled_contexts) + 1)
                    prefix = "\n\n".join(sampled_contexts[:idx]) + "\n\n"
                    suffix = "\n\n" + "\n\n".join(sampled_contexts[idx:])

                    new_context = prefix + context + suffix
                    example["context"] = new_context

                    for j, answer_start in enumerate(example["answers"]["answer_start"]):
                        example["answers"]["answer_start"][j] = len(prefix) + answer_start

                    f.write(json.dumps(example) + "\n")


def main(directory: str = "../data/squad_with_distracting_contexts"):
    write_squad(directory)


if __name__ == '__main__':
    Fire(main)
