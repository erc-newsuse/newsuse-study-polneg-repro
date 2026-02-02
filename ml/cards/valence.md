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
- name: erc-newsuse-valence
  results: []
---

# erc-newsuse-valence

This model is a fine-tuned version of [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) trained on a custom human-labelled dataset of social media posts published by major news media outlets in six countries: U.S., U.K., Ireland, Poland, France and Spain between 2020 and 2024.

## Model description

The purpose of the model is to discriminate between different kinds of valence in news articles, headlines, and posts.
To define valence, we followed the conceptual work by [Lengauer et al. (2011)](https://doi.org/10.1177/1464884911427800),
who proposed definitions and measurements of valence in news. Our operationalization and classifier define valence by distinguishing
between “the mere dissemination of negative news” (exogenous valence coming into the news from outside, that is, from the topic itself)
and “endogenous valence imposed on news by journalists through their usage of language”. More concretely, it assesses two aspects separately:

1. Event valence, or whether given news is likely to be perceived as negative, neutral, positive.
2. Sentiment, or whether language and general framing of the news is negative, neutral, positive.

## Intended uses & limitations

Research purposes, in particular selection of texts from large diverse corpora and/or calculation of statistics in groups (i.e. for negative and non-negative content). The design and conceptualization of this model was tailored for a specific research project and may not be relevant in other contexts.

The model should work well for the languages it was fined-tuned on. However, since it is based on a multilingual backbone it may also work relatively well for other languages. That said, in such cases a noticeable drop in performance is expected.

## Training and evaluation data

Cannot be shared for legal reasons. The scores obtained on a validation hold-out subset of the dataset were:

| F1(event) | F1(sentiment) |
| --------- | ------------- |
| 0.807     | 0.819         |

## Usage

The model uses custom processing and pipeline logic, so has to be initialized in a slightly more verbose way than standard models. For the same reason, it has to be loaded with the `trust_remote_code=True` flag. Last but not least, this model has two separate classification heads, and thus to be used as a `transformers` pipeline it must select a custom `text-multi-classification` task.

```python
from transformers import AutoModel, AutoTokenizer, pipeline

model = AutoModel.from_pretrained("sztal/erc-newsuse-valence", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model.config.base_name_or_path)
classifier = pipeline("text-multi-classification", model=model, tokenizer=tokenizer)
```

Now, the pipelines is ready to be applied to text data in the standard way.

```python
text = """
Bob Dole tributes pour in after former GOP Senator dies at 98
”Senator Dole was an American hero, a statesman of the highest order & one of the greatest legislators of all time,” wrote U.S.
Senator Roger Marshall.
""".strip()

classifier(text)
# {'event': {'label': -1, 'score': 0.6979869604110718},
#  'sentiment': {'label': 1, 'score': 0.5554186701774597}}
```

### Framework versions

- `python==3.12`
- `numpy==2.3.5`
- `scipy==1.16.2`
- `torch==2.9.0`
- `transformers==4.45.2`
