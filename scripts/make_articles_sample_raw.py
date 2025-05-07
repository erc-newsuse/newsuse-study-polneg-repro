# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import config, paths

# %% ---------------------------------------------------------------------------------

dataset = DataFrame.from_(paths.dataset)

# %% ---------------------------------------------------------------------------------

keys = ["country", "name", "political", "negativity"]
sample = (
    dataset.query("type.eq('link')")[["key", *keys, "post_url"]]
    .groupby(keys)
    .apply(
        lambda df: df.sample(
            n=min(config.articles_sample.groupsize, len(df)), random_state=3031
        ),
        include_groups=False,
    )
    .reset_index(level=list(range(len(keys))), drop=False)
    .reset_index(drop=True)
)

# %% -------------------------------------------------------------------------------

keys = ["country", "political", "negativity"]
small_sample = (
    sample.groupby(keys)
    .sample(n=config.articles_sample.groupsize, random_state=30317)
    .reset_index(drop=True)
)

sample["small"] = sample["key"].isin(small_sample["key"])

# %% ---------------------------------------------------------------------------------

sample.to_(paths.articles_sample)

# %% ---------------------------------------------------------------------------------
