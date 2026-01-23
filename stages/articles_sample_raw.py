# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import config, paths

# %% ---------------------------------------------------------------------------------

dataset = DataFrame.from_(paths.final)

# %% ---------------------------------------------------------------------------------

groups = ["country", "name", "political", "valence"]
sample = (
    dataset.merge(DataFrame.from_(paths.dataset, columns=["key", "type", "post_url"]))
    .query("type.eq('link')")[["key", *groups, "event", "sentiment", "post_url"]]
    .groupby(groups)
    .apply(
        lambda df: df.sample(
            n=min(config.articles_sample.groupsize, len(df)), random_state=3031
        ),
        include_groups=False,
    )
    .reset_index(level=list(range(len(groups))), drop=False)
    .reset_index(drop=True)
)

# %% -------------------------------------------------------------------------------

groups = ["country", "political", "valence"]
small_sample = (
    sample.groupby(groups)
    .sample(n=config.articles_sample.groupsize, random_state=30317)
    .reset_index(drop=True)
)

sample["small"] = sample["key"].isin(small_sample["key"])

# %% ---------------------------------------------------------------------------------

sample.to_(paths.articles_sample)

# %% ---------------------------------------------------------------------------------
