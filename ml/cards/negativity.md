---
library_name: transformers
license: mit
base_model: FacebookAI/xlm-roberta-large
language:
- en
- pl
- fr
- es
pipeline_tag: text-classification
model-index:
- name: erc-newsuse-negativity
  results: []
---

# erc-newsuse-negativity

This model is a fine-tuned version of [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) trained on a custom human-labelled dataset
of social media posts published by major news media outlets in six countries: U.S., U.K., Ireland, Poland, France and Spain between 2020 and 2024.

## Model description

It was fined-tuned for the purpose of discrimination (binary classification) between negative and non-negative news posts.
To define negativity, we followed the conceptual work by [Lengauer et al. (2011)](https://doi.org/10.1177/1464884911427800),
who proposed definitions and measurements of negativity in news. Our operationalization and classifier define negativity by combining
“the mere dissemination of negative news” (exogenous negativity coming into the news from outside, that is, from the topic itself)
and “endogenous negativity imposed on news by journalists through their usage of language”.

In particular, the classifier should be sensitive to both negative sentiment expressed
through the use of language, as well as coverage of negative events such as:
- crimes
- accidents and disasters
- wars and clashes
- major disruptions of social life and/or order (e.g. COVID, major protests)


## Intended uses & limitations

Research purposes, in particular selection of texts from large diverse corpora and/or calculation of statistics in groups (i.e. for negative and non-negative content).
The design and conceptualization of this model was tailored for a specific research project and may not be relevant in other contexts.
In particular, users should be aware of the specific definition of "negative" assumed by this classifier.

The model should work well for the languages it was fined-tuned on.
However, since it is based on a multilingual backbone it may also work relatively well for other languages.
That said, in such cases a noticeable drop in performance is expected.

## Training and evaluation data

Cannot be shared for legal reasons. The scores obtained on a validation hold-out subset of the dataset were:

| F1(negative)  | F1(other) |
| ------------- | --------- |
| 0.915         | 	0.908   |

## Usage

The easiest way to apply the model in practice is to load it as a text classification pipeline.

```python
from transformers import pipeline

classifier = pipeline("text-classification", "sztal/erc-newsuse-negativity")
```

### Examples

```python
negative_texts = [
    'Minnesota police officer will be charged with second-degree manslaughter in the shooting of Daunte Wright during a traffic stop on Sunday',
    "Ghost guns don't have serial numbers and are assembled from parts that can be ordered online. Last year, as the pandemic coincided with a spike in gun purchases, ghost guns were found at an increasing rate in cities across the U.S. Deadly and Untraceable, ‘Ghost Guns’ Are Becoming More Common in N.Y.",
    'A National Transportation Safety Board team was planning to start work at the scene of a deadly highway crash in Ohio involving a charter bus filled with high school students that left six people dead and 18 injured.',
    'The United States recorded its 12th million COVID-19 case on Saturday, even as millions of Americans were expected to travel for the upcoming Thanksgiving holiday, ignoring warnings from health officials about furthering the spread of the infectious disease.',
]

classifier(negative_texts)
# [{'label': 'NEGATIVE', 'score': 0.9830681681632996},
#  {'label': 'NEGATIVE', 'score': 0.904019832611084},
#  {'label': 'NEGATIVE', 'score': 0.9729166626930237},
#  {'label': 'NEGATIVE', 'score': 0.8813401460647583}]
```

```python
other_texts = [
    "A large crowd gathered in Khost on August 31, waving Talban flags and hoisting coffins draped with the US, UK, and French flags aloft. Photos show Taliban supporters holding a mock funeral for the US and UK, parading makeshift coffins draped with the countries' flags",
    '"At some point, an emergency stops being an emergency and instead becomes ... life." Has COVID hit that point with Omicron?',
    'In an effort to battle the staffing shortage, the Departments of Transportation and Education will allow states to waive portions of the applicant test. School bus driver shortage could lead to less knowledgeable drivers'
]

classifier(other_texts)
# [{'label': 'OTHER', 'score': 0.9389849305152893},
#  {'label': 'OTHER', 'score': 0.9963881969451904},
#  {'label': 'OTHER', 'score': 0.9793765544891357}]
```

## Training procedure

Standard training loop using [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer).

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-06
- train_batch_size: 16
- eval_batch_size: 8
- seed: 1884749421
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.25
- num_epochs: 10
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.45.2
- Pytorch 2.6.0+cu124
- Datasets 3.5.0
- Tokenizers 0.20.3
