# %% Setup ---------------------------------------------------------------------------
library(reticulate)
library(arrow)
library(dplyr)
library(tibble)
library(glmmTMB)

use_python(normalizePath(R.home("../../bin/python")), required = TRUE)

project <- import("project")
config  <- project$config
paths   <- project$paths

dirpath <- paths$glmm / "valence"
dirpath$mkdir(parents = TRUE, exist_ok = TRUE)

countries <- config$countries$order
countries <- countries[0L:length(countries)]

# %% Get data ------------------------------------------------------------------------

dataset <- as.character(paths$dataset) %>%
    read_parquet %>%
    tibble %>%
    mutate(
        country = factor(country, levels = countries),
        political = factor(political, levels = c("OTHER", "POLITICAL")),

        valence = factor(valence, levels = c("OTHER", "NEGATIVE")),
        year = as.factor(year),
        month = as.factor(month),
        day = as.factor(day),
        weekday = as.factor(weekday),
    )

# %% Define fitting function ---------------------------------------------------------

fitglmm <- function(formula, data, control = list(), parallel = list(), ...) {
    ncores   <- min(parallel::detectCores(), 8L)
    parallel <- rlang::ll(n = ncores, autopar = TRUE, !!!parallel)
    control  <- rlang::ll(profile = TRUE, parallel = parallel, !!!control)
    glmm <- glmmTMB(
        formula, data, family = binomial,
        control = do.call(glmmTMBControl, control), ...
    )
    glmm
}

# %% Define model data ---------------------------------------------------------------

data <- dataset %>%
    # sample_n(100000L, replace = FALSE) %>%
    tibble

# %% Fit model 0 ---------------------------------------------------------------------

frm <- frm0 <- valence ~ country * political +
    (1 + political | country:name) + (1 + political | country:year:month:day)
time <- time0 <- system.time(glmm <- fitglmm(frm, data))
print(time)

saveRDS(glmm, as.character(dirpath / "0.rds"), compress = TRUE)
rm(glmm)
gc()

# %% ---------------------------------------------------------------------------------
