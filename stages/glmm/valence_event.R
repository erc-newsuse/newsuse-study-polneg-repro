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

frm <- event ~ country * political +
    (1 + political | country:name) + (1 + political | country:year:month:day)

# %% ---------------------------------------------------------------------------------

system.time(
    glmm <- brm(
        formula = frm,
        data = data,
        family = cumulative(link = "logit"),
        cores = min(parallel::detectCores() - 2L, 16L),
        iter = 2000L,
        algorithm = "sampling",
        prior = c(
            prior(normal(0, 1), class = "sd", group = "country:name"),
            prior(normal(0, 1), class = "sd", group = "country:year:month:day")
        )
        # warmup = 1000L,
        # control = list(adapt_delta = 0.95, max_treedepth = 15L)
    )
)

# %% ---------------------------------------------------------------------------------
