# %% Setup ---------------------------------------------------------------------------
library(reticulate)
library(arrow)
library(dplyr)
library(tibble)
library(brms)

use_python(normalizePath(R.home("../../bin/python")), required = TRUE)

project <- import("project")
config  <- project$config
paths   <- project$paths

dirpath <- paths$glmm / "valence"
dirpath$mkdir(parents = TRUE, exist_ok = TRUE)

countries <- config$countries$order
countries <- countries[0L:length(countries)]

# %% Get data ------------------------------------------------------------------------

data <- as.character(paths$final) %>%
    read_parquet %>%
    mutate(
        country = factor(country, levels = countries),
        political = factor(political, levels = c("OTHER", "POLITICAL")),
        event = factor(event, ordered = TRUE),
        sentiment = factor(sentiment, ordered = TRUE),
        valence = factor(valence, ordered = TRUE),
        year = as.factor(year),
        month = as.factor(month),
        day = as.factor(day),
    )
# data <- sample_n(data, 1000000L)

# %% ---------------------------------------------------------------------------------

frm <- sentiment ~ country * political +
    (1 | country:name:political) + (1 | country:year:month:day:political)
    # (1 + political | country:name) + (1 + political | country:year:month:day)

# %% ---------------------------------------------------------------------------------

# Note: glmmTMB does not support cumulative link models for ordinal data (only ordbeta).
# We use brms with cmdstanr backend for efficiency.

system.time(
    glmm <- brm(
        formula = frm,
        data = data,
        family = cumulative(link = "logit"),
        backend = "cmdstanr",
        threads = threading(4), # Adjust based on available cores (chains * threads <= total cores)
        cores = 4,
        iter = 2000L,
        prior = c(
            prior(
                normal(0, 1.253314), lb = 0, class = "sd",
                # group = "country:name"
                group = "country:name:political"
            ),
            prior(
                normal(0, 1.253314), lb = 0, class = "sd",
                # group = "country:year:month:day"
                group = "country:year:month:day:political"
            )
        ),
        # algorithm = "pathfinder",
        algorithm = "meanfield",
        control = list(refresh = 5L)
        # control = list(adapt_delta = 0.95, max_treedepth = 15L)
    )
)

# %% ---------------------------------------------------------------------------------
