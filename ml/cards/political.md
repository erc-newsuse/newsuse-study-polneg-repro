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
- name: erc-newsuse-political
  results: []
---

# erc-newsuse-political

This model is a fine-tuned version of [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) trained on a custom human-labelled dataset
of social media posts published by major news media outlets in six countries: U.S., U.K., Ireland, Poland, France and Spain between 2020 and 2024.

## Model description

It was fined-tuned for the purpose of discrimination (binary classification) between content related to, broadly defined, political issues, and other non-political content.
It uses a broad definition of what counts as political, following some previous works in communication studies such as [Wojcieszak et al. (2023)](https://doi.org/10.1080/10584609.2023.2238641).
Namely, this classifier conceptualizes "politics" rather broadly:
> including references to both political figures, policies, elections, news events (e.g., impeachment inquiry, the primaries)
> as well as issues such as climate change, immigration, healthcare, gun control, sexual assault, racial, gender, sexual, ethnic, and religious minorities, the regulation of large tech companies, and crimes involving guns.

## Intended uses & limitations

Research purposes, in particular selection of texts from large diverse corpora and/or calculation of statistics in groups (i.e. for political and non-political content).
The design and conceptualization of this model was tailored for a specific research project and may not be relevant in other contexts.
In particular, users should be aware of the broad definition of "political" assumed by this classifier.

The model should work well for the languages it was fined-tuned on.
However, since it is based on a multilingual backbone it may also work relatively well for other languages.
That said, in such cases a noticeable drop in performance is expected.

## Training and evaluation data

Cannot be shared for legal reasons. The scores obtained on a validation hold-out subset of the dataset were:

| F1(political) | F1(other) |
| ------------- | --------- |
| 0.889         | 	0.907   |

## Usage

The easiest way to apply the model in practice is to load it as a text classification pipeline.

```python
from transformers import pipeline

classifier = pipeline("text-classification", "sztal/erc-newsuse-political")
```

### Examples

```python
political_texts = [
    'Greene recently chased Ocasio-Cortez down a hallway as the two left the House chamber, shouted at her, and accused her of supporting terrorists.',
    'The ex-president will make his first big speech since leaving the White House at the conference.',
    'Employers continue to fight to retain workers amid a tight labor market and growing Omicron coronavirus variant concerns.'
]

classifier(political_texts)
# [{'label': 'POLITICAL', 'score': 0.9945843815803528},
#  {'label': 'POLITICAL', 'score': 0.9939272403717041},
#  {'label': 'POLITICAL', 'score': 0.9750990271568298}]
```

```python
other_texts = [
    'A dental surgery student has turned heads for her viral video claiming that she and other dentists known when women are pregnant by the state of their teeth and gums.',
    '"I was right at her door, about to leave. And for some reason, she just asked me to stay." Resident of collapsed Florida building says he\'s alive only because girlfriend persuaded him to stay with her',
    'I am destroyed. I do not feel good," Hamilton said after finishing third in Sunday\'s Abu Dhabi Grand Prix.',
]

classifier(other_texts)
# [{'label': 'OTHER', 'score': 0.8563344478607178},
#  {'label': 'OTHER', 'score': 0.9842121005058289},
#  {'label': 'OTHER', 'score': 0.9840729832649231}]
```

```python
# Here is a borderline text that gets classified as 'POLITICAL', but with low certainty
borderline_text = "As the race for three casino licenses in the New York City region kicks off in earnest this year, developers have launched charm offensives to gain public support. Here are their proposals and the most likely casino sites."
classifier(borderline_text)
# [{'label': 'POLITICAL', 'score': 0.5392860174179077}]
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
