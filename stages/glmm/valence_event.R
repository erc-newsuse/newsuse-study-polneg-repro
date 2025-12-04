# %% Setup ---------------------------------------------------------------------------
library(reticulate)
library(stringr)
library(arrow)
library(dplyr)
library(tibble)
library(brms)

use_python(normalizePath(R.home("../../bin/python")), required = TRUE)
builtins <- import("builtins")

project <- import("project")
config  <- project$config
paths   <- project$paths

dirpath <- paths$glmm / "valence"
dirpath$mkdir(parents = TRUE, exist_ok = TRUE)

countries <- names(builtins$dict(config$categorical$countries))

target <- "event"
opts   <- config$glmm$valence[[target]]

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

# %% ---------------------------------------------------------------------------------

formula <- opts$formula %>%
    str_glue %>%
    str_replace_all("[\n\\s]+", " ") %>%
    as.formula

# %% ---------------------------------------------------------------------------------

fit <- function(formula, data, seed = NULL, ...) {
    opts <- rlang::ll(
        formula = formula, data = data, seed = seed,
        !!!rlang::ll(
            family = cumulative(link = opts$link),
            algorithm = opts$algorithm,
            backend = opts$backend,
            chains = opts$chains,
            cores = opts$cores,
            threads = threading(n_threads),
            iter = opts$iter,
            # prior = c(
            #     prior(normal(0, 1.253314), lb = 0, class = "sd")
            # ),
        )
    )
    do.call(brm, opts)
}

# %% ---------------------------------------------------------------------------------

# Note: glmmTMB does not support cumulative link models for ordinal data (only ordbeta).
# We use brms with cmdstanr backend for efficiency.

system.time(
    glmm <- fit(formula, data, seed = opts$seed)
)

saveRDS(
    glmm,
    as.character(dirpath / str_glue("{target}.rds")),
    compress = TRUE
)

# %% ---------------------------------------------------------------------------------
