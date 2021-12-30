# Experiments

## Motivation

The QuALITY dataset is very different from reading comprehension datasets like SQuAD. For example, the domain is very
different, the annotators had very different instructions and rewards, and the length of the passages is very long.

Therefore, it is not clear what causes models to perform better or worse on this dataset compared to SQuAD. Do they
struggle with what or why questions? Or do they struggle with the complexity of the questions? Or it is purely the need
to operate above long sequences?

## SQuAD with distracting contexts

In this experiment we check whether long contexts are enough to significantly degrade models' performance by adding
distracting contexts that inflate the context length. Specifically, we take all SQuAD passages and for each example we
sample additional passages aside from the original one. Then, we mix the passages to create a new inflated context.

Additionally, since the QuALITY has only 2523 example we limit check the performance of our models with a different
number of training examples. Currently, we do not sample using different random seeds, but take the first examples.

### Example

_Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?_

_Original Context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden
statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with
arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart.
Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at
Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main
drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of
Mary._

_Context + 1 distractor: In the years that followed, Eisenhower increased the number of U.S. military advisors in South
Vietnam to 900 men. This was due to North Vietnam's support of \"uprisings\" in the south and concern the nation would
fall. In May 1957 Diem, then President of South Vietnam, made a state visit to the United States for ten days. President
Eisenhower pledged his continued support, and a parade was held in Diem's honor in New York City. Although Diem was
publicly praised, in private Secretary of State John Foster Dulles conceded that Diem had been selected because there
were no better alternatives._

_Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the
Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised
with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately
behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes,
France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and
in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."_

### Running the experiment with 0, 16, and 64 distractors and 1024, 4096, 16384, and 65536 examples.

```bash
# create data
rm -rf data/squad_with_distracting_contexts
cd scripts
python create_squad_with_distractor_contexts.py
cd ..

# send experiments to slurm
bash scripts/run_squad_with_distracting_contexts.sh
```

### Results

As can be seen in
the [Wandb logs](https://wandb.ai/yuvalkirstain/squad_with_distracting_contexts?workspace=user-yuvalkirstain)
it looks like adding more distracting contexts leads to inferior performance. However, as we add more examples...

## SQuAD with full context

In the last experiment we just added distracting contexts from other paragraphs. What will happen if we use the full
wikipedia article of the correct context? Will that make it harder for the model to understand what it does wrong? We
suspect that it will be harder for the model to perform with the original document, since the distractors might be
harder. However, we should note that there are other factors that might make this harder.

### Running the experiment with 1024, 4096, 16384, and 65536 examples.

```bash
bash scripts/run_squad_with_full_doc.sh
```

### Results

As can be seen in the [Wandb logs](https://wandb.ai/yuvalkirstain/squad_with_full_doc?workspace=user-yuvalkirstain) the
performance with 1024 examples is very low and models achieve about 50 EM with 1024 examples. However, as we increase
the size of the training set ...

## QuALITY as SQuAD

We will try to make a variation of QuALITY to evaluate it similarly as we do for SQuAD. This will allow us to operate on
the large sequences that can be found in the dataset.
