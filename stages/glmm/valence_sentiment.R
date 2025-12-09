# %% Setup ---------------------------------------------------------------------------
library(reticulate)
library(stringr)
library(arrow)
library(dplyr)
library(tibble)
library(brms)
library(purrr)

use_python(normalizePath(R.home("../../bin/python")), required = TRUE)
builtins <- import("builtins")

project <- import("project")
config  <- project$config
paths   <- project$paths

dirpath <- paths$glmm / "valence"
dirpath$mkdir(parents = TRUE, exist_ok = TRUE)

countries <- names(builtins$dict(config$categorical$countries))

target <- "sentiment"
opts   <- config$glmm$valence$targets[[target]]

n_threads <- min(opts$n_threads, parallel::detectCores() - 2L)

# %% Get data ------------------------------------------------------------------------

data <- as.character(paths$final) %>%
    read_parquet %>%
    mutate(
        country = factor(country, levels = countries),
        event = factor(event, ordered = TRUE),
        sentiment = factor(sentiment, ordered = TRUE),
        valence = factor(valence, ordered = TRUE),
    )

# %% Subsample data ------------------------------------------------------------------

subsample <- builtins$dict(opts)$subsample
if (!is.null(subsample)) {
    set.seed(opts$seed + 17L)
    data <- sample_n(data, subsample)
}

# %% ---------------------------------------------------------------------------------

formula <- opts$model$formula %>%
    str_glue %>%
    str_replace_all("[\n\\s]+", " ") %>%
    as.formula

# %% ---------------------------------------------------------------------------------

prior <- tryCatch(
    {
        map(builtins$list(opts$model$prior), ~do.call(set_prior, builtins$dict(.x))) %>%
            reduce(c)
    },
    error = function(e) NULL
)

# %% ---------------------------------------------------------------------------------

fit <- function(formula, data, seed = NULL, ...) {
    solver <- builtins$dict(opts$solver)
    solver$control <- builtins$dict(solver$control)
    opts <- rlang::ll(
        formula = formula,
        data = data,
        seed = seed,
        family = c(opts$model$family, opts$model$link),
        prior = prior,
        # threads = threading(n_threads),
        !!!solver,
    )
    do.call(brm, opts)
}

# %% ---------------------------------------------------------------------------------

# Note: glmmTMB does not support cumulative link models for ordinal data (only ordbeta).
# We use brms with cmdstanr backend for efficiency.

system.time(
    glmm <- fit(formula, data, seed = opts$seed)
)

# %% ---------------------------------------------------------------------------------

saveRDS(
    glmm,
    as.character(dirpath / str_glue("{target}.rds")),
    compress = TRUE
)

# %% ---------------------------------------------------------------------------------
