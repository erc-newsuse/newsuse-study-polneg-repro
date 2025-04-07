# Political negativity

This is a repository for reproducing the results from the paper

> Negative news is less prevalent and generates lower user engagement
> than non-negative news across six countries

## [Github repo](https://github.com/erc-newsuse/newsuse-study-polneg-repro)

Code (without data) is archived in a Github repository.

## Text data availability

For legal reasons we do not distribute the exact content of the posts.
However, in all cases metadata allowing for identification of and access to
individual posts is retained, so the dataset can be _rehydrated_ by filling
the missing data after obtaining the content, i.e. through Facebook API,
by scraping content of selected posts or obtaining it by any other means.
We can also share the content data for private use upon a reasonable request.

> **NOTE.** Most importantly, all analyses in this study, other than model training
> and post labeling, do not require text content and can be reproduced without it.


## [OSF data and code repository](https://osf.io/79a46/?view_only=f7cad71ef9b54a5a866e2e579eb92e62)

Dehydrated data and full code is archived as an OSF repository,
which allows for reproduction of all analyses other than the training of classifiers and posts' classification even without data rehydration.
It uses a simplified set of computation stages defined `dvc.yaml`
(run `dvc stage list` or `dvc dag` to see them) that uses a precomputed
set of post labels (in `data/proc/cls.parquet` file).

In order to reproduce the full pipeline for the data including post content
rename `dvc-fulldata.yaml` to `dvc.yaml` and run `dvc repro`.

> **NOTE.** Read _Reproduction_ section below for more details on how to run and inspect the pipeline.

### Key fields

There are two key fields allowing, in principle, for rehydration of the Facebook data:

1. `key` : it contains a unique Facebook post identifier (namespace prefix ending with `@`, i.e. `sotrender@`, needs to be removed to obatin the actual post ID)
2. `post_url` : stores URLs pointing to individual posts

> **NOTE.** In order to replicate both classifier training and post labeling the content data must be rehydrated both in the raw posts
> data files (`posts-eu.parquet` and `posts-us.parquet`) as well as
> in training datasets (`ml/classifiers/political` and `ml/classifiers/negativity`).

## Reproduction

The project management, including versioning of the most of important
software libraries, is handled using
[Conda package manager](https://anaconda.org/anaconda/conda).

### Project setup


```bash
conda env create -f environment.yaml
conda activate polneg-repro
make init
```

### Computation stages

The project workflow is organized using [DVC](https://dvc.org/),
which allows for defining all important computation stages as
interdependent tasks.

```bash
dvc stage list  # list defined computation stages
```

This should be the output:

```
data                        Outputs data/proc/fulldata.parquet, data/proc/textdata.parquet
train@political             Outputs ml/classifiers/political
train@negativity            Outputs ml/classifiers/negativity
labels@political            Outputs data/proc/cls-political.parquet
labels@negativity           Outputs data/proc/cls-negativity.parquet
merge-labels                Outputs data/proc/cls.parquet
daily-counts                Outputs data/proc/daily.parquet
dataset                     Outputs data/proc/dataset.parquet, data/proc/quality.parquet
glmm@negativity             Outputs models/glmm/negativity
glmm@likes                  Outputs models/glmm/likes
glmm@comments               Outputs models/glmm/comments
glmm@shares                 Outputs models/glmm/shares
```

One can also display a directed acyclic graph of relationships between the tasks.

```bash
dvc dag
```

All analyses used in the paper can be reproduced simply by running the
following command:


```bash
dvc repro
```

> **IMPORTANT.** A full run of the entire pipeline may take several hours
> or even more than a day of compute time

### Running individual stages

Individual computation stages
(of which names are given by `dvc stage list`)
can be run simply using `dvc repro <name>`, where `<name>` is one or
more stage names (separated by spaces).

## Main results

The pipeline run by `dvc repro` converts raw data into final dataset(s)
used in the paper, fits generalized linear mixed effects models,
and possibly also run the training of machine learning
classifiers and assigns classification labels to posts.
In other words, it runs all the time-consuming computations.

The actual analyses and plots from the paper are organized
as [Quarto](https://quarto.org/) notebooks, which allow for easy mixing of Python
and R code. Thus, results and figures can be reproduced by rerunning the code
from the notebooks.

> **NOTE.** We used [matplotlib](https://matplotlib.org/) with TeX-based text rendering
> for generating the figures. This allows for beautiful typesetting but may makes
> reproducibility more difficult. In case of problems try switching
> `plotting.usetex` param in `params.yaml` to `false`.

In particular there are the following notebooks and scripts in the `analyses` subfolder:

```
analyses/
├── classifiers              // python scripts for evaluating classifier performance
│   ├── negativity.py
│   └── political.py
├── descriptives
│   ├── descriptives.py      // descriptive statistics
│   └── examples.py          // examples of text classifications
├── glmm
│   ├── comments
│   │   └── validation.qmd   // validation of GLMM for comments
│   ├── likes
│   │   └── validation.qmd   // validation of GLMM for likes
│   ├── negativity
│   │   └── validation.qmd   // validation of GLMM for negativity
│   ├── shares
│   │   └── validation.qmd   // validation of GLMM for shares
│   ├── models-table.qmd     // generation of the model coefficient table
│   ├── polneg.qmd           // generation of negativity prevalence subfigure
│   ├── engagement.qmd       // generation of engagement subfigure
│   └── simulation.qmd       // generation of figure with simulation results
```


## Classifiers

Classifiers are stored in `classifiers` folder. They are saved using standard format
used by `transformers` library from [HuggingFace](https://huggingface.co/),
and should be compatible with `transformers>=4.44` and `torch>=2.4`.
See the `scripts/make_labels.py` script for an example.

They are also accessible through _Huggingface Hub_ [here](https://huggingface.co/sztal/erc-newsuse-political)
and [there](https://huggingface.co/sztal/erc-newsuse-negativity).
