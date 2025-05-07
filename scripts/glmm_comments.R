# %% Setup ---------------------------------------------------------------------------
library(reticulate)
library(arrow)
library(dplyr)
library(tibble)
library(glmmTMB)

use_condaenv("polneg-repro")

project <- import("project")
config  <- project$config
paths   <- project$paths

dirpath <- paths$glmm / "comments"
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
        negativity = factor(negativity, levels = c("OTHER", "NEGATIVE")),
        year = as.factor(year),
        month = as.factor(month),
        day = as.factor(day),
        weekday = as.factor(weekday),
    )

# %% Define model data ---------------------------------------------------------------

data <- dataset %>%
    # sample_n(10000L, replace = FALSE) %>%
    tibble

# %% Define fitting function ---------------------------------------------------------

fitglmm <- function(
    formula,
    data,
    family = nbinom2,
    ziformula = ~1,
    control = list(),
    parallel = list(),
    ...
) {
    ncores   <- min(parallel::detectCores(), 8L)
    parallel <- rlang::ll(n = ncores, autopar = TRUE, !!!parallel)
    control  <- rlang::ll(profile = TRUE, parallel = parallel, !!!control)
    glmm <- glmmTMB(
        formula, data, family = family, ziformula = ziformula,
        control = do.call(glmmTMBControl, control), ...
    )
    glmm
}

# %% Fit model 0 ---------------------------------------------------------------------

frm <- frm0  <- comments ~ country * political * negativity +
    (1 + political * negativity | country:name) +
    (1 + political * negativity | country:year:month:day)
zfrm <- ~0
dfrm <- ~country + (1 | country:year:month:day)
time <- time0 <- system.time(
    glmm <- fitglmm(frm, data, ziformula = zfrm, dispformula = dfrm)
)
print(time)

saveRDS(glmm, as.character(dirpath / "0.rds"))
rm(glmm)
gc()

# %% ---------------------------------------------------------------------------------
