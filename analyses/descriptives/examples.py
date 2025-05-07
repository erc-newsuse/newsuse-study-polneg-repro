# %%
from newsuse.data import DataFrame

from project import config, paths

countries = config.countries.labels

data = DataFrame.from_(paths.dataset)[["key", "country", "political", "negativity"]].merge(
    DataFrame.from_(paths.text), on="key", how="left"
)

# %%
examples = (
    data.groupby(["country", "political", "negativity"])
    .sample(5, random_state=117171)
    .reset_index(drop=True)
)

# %% US
examples.pipe(
    lambda df: print(
        df.set_index("country")
        .loc[["us", "uk"]]
        .set_index(["political", "negativity"], append=True)
        .rename(countries, level="country")
        .rename({"POLITICAL": "political", "OTHER": "non-political"}, level="political")
        .rename({"NEGATIVE": "negative", "OTHER": "non-negative"}, level="negativity")
        .style.hide(["key"], axis="columns")
        .format(escape="latex")
        .to_latex(
            multicol_align="c",
            multirow_align="t",
            hrules=True,
        )
    )
)

# %%
