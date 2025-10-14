# %% ---------------------------------------------------------------------------------

library(dplyr)
library(purrr)
library(glmmTMB)
library(reticulate)
library(arrow)
library(lubridate)
library(stringr)

use_condaenv("polneg-repro")

project <- import("project")
config  <- project$config
paths   <- project$paths

# %% ---------------------------------------------------------------------------------

dataset <- as.character(paths$dataset) %>%
    read_parquet() %>%
    select(country, name, political, negativity, year, month, day, timestamp)

gc()

# %% ---------------------------------------------------------------------------------

start <- min(dataset$timestamp)
end   <- max(dataset$timestamp)
time  <- seq.POSIXt(start, end, by = "day")

outlets    <- unique(with(dataset, interaction(country, name, sep = ":")))
political  <- unique(dataset$political)
negativity <- unique(dataset$negativity)

grid <- do.call(expand.grid, rlang::ll(
    outlet = outlets,
    political = political,
    negativity = negativity,
    timestamp = time,
)) %>%
    transmute(
        country = str_split_i(outlet, ":", 1L),
        name = str_split_i(outlet, ":", 2L),
        political = political,
        negativity = negativity,
        year = year(timestamp),
        month = month(timestamp),
        day = day(timestamp),
    ) %>%
    tibble

# %% ---------------------------------------------------------------------------------

metrics <- c("reactions", "comments", "shares")

# %% ---------------------------------------------------------------------------------

predictions <- map(metrics, ~{
    glmm <- readRDS(as.character(paths$glmm / .x / "0.rds"))
    pred <- predict(glmm, grid, type = "response", allow.new.levels = TRUE)
    gc()
    mutate(grid, metric = .x, prediction = pred)
}) %>% bind_rows

# %% ---------------------------------------------------------------------------------

write_parquet(
    predictions,
    as.character(paths$predictions),
    compression = "zstd",
    compression_level = 9,
)

# %% ---------------------------------------------------------------------------------
