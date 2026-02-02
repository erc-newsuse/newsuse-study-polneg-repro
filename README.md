# Political negativity

This is a repository for reproducing the results from the paper:

> Limited news negativity on Facebook: Evidence on prevalence and engagement from six countries


## Project structure

```
├── stages/              # Pipeline scripts (Python)
│   ├── ml/              # Machine learning classification stages
│   ├── glmm/            # Generalized linear mixed models
│   ├── analysis/        # Analysis and visualization stages
│   └── descriptives/    # Descriptive statistics stages
├── project/             # Local Python package with shared utilities
├── analyses/            # Quarto notebooks and scripts for reporting
├── data/                # DVC-managed data storage
│   ├── raw/             # Raw input data
│   ├── proc/            # Processed data
│   └── aux/             # Auxiliary data
├── glmm/                # Fitted GLMM model outputs (NetCDF format)
├── ml/                  # Machine learning resources
├── figures/             # Generated figures
└── tables/              # Generated tables
```


## Reproduction

The project management, including versioning of the most important
software libraries, is handled using
[Conda package manager](https://anaconda.org/anaconda/conda).

### Project setup

```bash
conda env create -f environment.yaml
conda activate newsuse-study-polneg
make init
# The last command is defined in `Makefile` and initializes
# the state of the environment, including installing all
# required Python and R packages and setting up GIT and DVC.
```

### Computation stages

The project workflow is organized using [DVC](https://dvc.org/),
which allows for defining all important computation stages as
interdependent tasks. All stages are defined in `dvc.yaml`
and implemented in the `stages/` directory.

```bash
dvc stage list  # list defined computation stages
dvc dag         # display directed acyclic graph of stage dependencies
```

#### Data processing stages

| Stage | Script | Description |
|-------|--------|-------------|
| `posts` | `stages/posts.py` | Process raw posts data into cleaned format |
| `daily-counts` | `stages/daily_counts.py` | Compute daily post counts per outlet |
| `dataset` | `stages/dataset.py` | Create main analysis dataset |
| `outlet-meta` | `stages/outlet_meta.py` | Process outlet metadata |
| `merge-cls` | `stages/merge_cls.py` | Merge classification labels |
| `final` | `stages/final.py` | Create final analysis-ready dataset |

#### Classification stages

| Stage | Script | Description |
|-------|--------|-------------|
| `political-classify` | `stages/ml/political_classify.py` | Classify posts as political/non-political |
| `valence-classify` | `stages/ml/valence_classify.py` | Classify post valence (negative/neutral/positive) |

#### GLMM stages

| Stage | Script | Description |
|-------|--------|-------------|
| `valence-glmm` | `stages/glmm/valence.py` | Fit valence prevalence models (event, sentiment, structural) |
| `valence-glmm-by` | `stages/glmm/valence_by.py` | Fit valence models by outlet quality/ideology |
| `valence-validation` | `stages/glmm/valence_validation.py` | Validate valence models |
| `engagement-glmm` | `stages/glmm/engagement.py` | Fit engagement models (reactions, comments, shares) |
| `engagement-glmm-by` | `stages/glmm/engagement_by.py` | Fit engagement models by outlet quality/ideology |
| `engagement-validation` | `stages/glmm/engagement_validation.py` | Validate engagement models |
| `ppd` | `stages/glmm/ppd.py` | Compute posterior predictive distributions |

#### Analysis stages

| Stage | Script | Description |
|-------|--------|-------------|
| `descriptives` | `stages/descriptives/valence.py`, `stages/descriptives/engagement.py` | Generate descriptive statistics |
| `valence-analysis` | `stages/analysis/valence.py` | Valence prevalence analysis |
| `valence-analysis-joint` | `stages/analysis/valence_joint.py` | Joint valence analysis |
| `engagement-analysis` | `stages/analysis/engagement.py` | Engagement analysis |
| `engagement-analysis-effects` | `stages/analysis/engagement_effects.py` | Engagement effect sizes |

### Running the pipeline

All analyses used in the paper can be reproduced by running:

```bash
dvc repro
```

> **IMPORTANT.** A full run of the entire pipeline may take several hours
> or even several days of compute time.

### Running individual stages

Individual computation stages can be run using `dvc repro <name>`,
where `<name>` is one or more stage names (separated by spaces).

To run a single stage without running all preceding stages:

```bash
dvc repro --single-item <name>
```


## Analyses and reporting

The pipeline run by `dvc repro` converts raw data into final datasets,
fits generalized linear mixed effects models, runs machine learning
classifiers, and assigns classification labels to posts.

The actual analyses and plots from the paper are organized
as Python scripts or [Quarto](https://quarto.org/) notebooks
in the `analyses/` directory:

```
analyses/
├── classifiers/
│   ├── political/
│   │   ├── iaa.py               # Inter-annotator agreement
│   │   └── performance.py       # Classifier performance metrics
│   ├── valence/
│   │   ├── iaa.py               # Inter-annotator agreement
│   │   └── performance.py       # Classifier performance metrics
│   ├── examples.py              # Classification examples
│   └── post_article_consistency.py
├── glmm/
│   ├── comments/
│   │   └── validation.qmd       # GLMM validation for comments
│   ├── reactions/
│   │   └── validation.qmd       # GLMM validation for reactions
│   ├── shares/
│   │   └── validation.qmd       # GLMM validation for shares
│   ├── engagement.qmd           # Engagement analysis figure
│   ├── models-table.qmd         # Model coefficients table
│   ├── outlets.qmd              # Outlet-level analysis
│   ├── polneg.qmd               # Valence prevalence figure
│   └── simulation.qmd           # Simulation results figure
```


## Classifiers

Classifiers are accessible through _Huggingface Hub_:

- [political](https://huggingface.co/sztal/erc-newsuse-political)
- [valence](https://huggingface.co/sztal/erc-newsuse-valence)

They use the standard format from the `transformers` library
and are compatible with `transformers>=4.45` and `torch>=2.4`.
