# %% Setup ---------------------------------------------------------------------------
library(reticulate)
library(stringr)
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

n_cores <- parallel::detectCores() - 2L
n_chains <- 4L
n_chain_threads <- min(4L, max(1L, floor(n_cores / n_chains)))

target <- "sentiment"

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

# %% ---------------------------------------------------------------------------------

frm <- as.formula(
    str_c(
        str_glue("{target} ~ country * political + "),
        "(1 + political || country:name) + (1 + political || country:year:month:day)"
    )
)

# %% ---------------------------------------------------------------------------------

fit <- function(formula, data, algortihm, seed = NULL, ...) {
    opts <- rlang::ll(
        formula = formula, data = data, algorith = algorithm, seed = seed,
        !!!rlang::ll(
            family = cumulative(link = "logit"),
            backend = "cmdstanr",
            chains = n_chains,
            cores = min(n_chains, n_cores),
            threads = threading(n_chain_threads),
            iter = 2000L,
            # prior = c(
            #     prior(normal(0, 1.253314), lb = 0, class = "sd")
            # ),
        )
    )
    do.call(brm, opts)
}

make_ppd <- function(model, newdata, ...) {
    opts <- rlang::ll(
        model, newdata = select(newdata, -n),
        !!!rlang::ll(
            ndraws = 100L,
            cores = min(n_chains * 2, n_cores)
        )
    )
    ppd <- do.call(posterior_predict, opts) - 2L
    pat <- "^V\\d+$"
    bind_cols(newdata, as_tibble(t(ppd))) %>%
        rowwise %>%
        mutate(ppd = list(c_across(matches(pat)))) %>%
        select(-matches(pat))
}

# %% ---------------------------------------------------------------------------------

# Note: glmmTMB does not support cumulative link models for ordinal data (only ordbeta).
# We use brms with cmdstanr backend for efficiency.

algorithm <- "meanfield"
system.time(
    glmm_mf <- glmm <- fit(
        formula = frm,
        data = data,
        algortihm = algorithm,
        seed = 432487L
    )
)

saveRDS(
    glmm_mf,
    as.character(dirpath / str_glue("{target}-{algorithm}.rds")),
    compress = TRUE
)

# %% ---------------------------------------------------------------------------------

agg <- tibble(glmm$data) %>%
    select(1L:day) %>%
    group_by(across(!(!!target))) %>%
    summarize(n = n()) %>%
    ungroup

ppd <- ppd_mf <- make_ppd(model = glmm_mf, newdata = agg)

write_parquet(
    ppd_mf,
    as.character(dirpath / str_glue("{target}-{algorithm}-ppd.parquet"))
)

# %% ---------------------------------------------------------------------------------

algorithm <- "fullrank"
system.time(
    glmm_fr <- glmm <- fit(
        formula = frm,
        data = data,
        algortihm = algorithm,
        seed = 303L
    )
)

saveRDS(
    glmm_fr,
    as.character(dirpath / str_glue("{target}.rds")),
    compress = TRUE
)

# %% ---------------------------------------------------------------------------------

ppd <- ppd_fr <- make_ppd(model = glmm_fr, newdata = agg)

write_parquet(
    ppd_fr,
    as.character(dirpath / str_glue("{target}-ppd.parquet"))
)

# %% ---------------------------------------------------------------------------------
