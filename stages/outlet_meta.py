# %% ---------------------------------------------------------------------------------

from newsuse.data import DataFrame

from project import paths

# %% ---------------------------------------------------------------------------------

sources = DataFrame.from_(paths.dataset, columns=["country", "name"]).drop_duplicates(
    ignore_index=True
)

# %% ---------------------------------------------------------------------------------

meta = DataFrame.from_(paths.raw / "outlet-meta.xlsx")[
    ["Unnamed: 0", "Unnamed: 1", "bias_trichotomized", "Quality_trichotomized"]
]
meta.columns = ["country", "name", "ideology", "quality"]

meta = meta.replace(
    {
        "country": {
            "United States": "us",
            "United Kingdom": "uk",
            "Ireland": "irl",
            "Poland": "pl",
            "France": "fr",
            "Spain": "esp",
        }
    }
)

# %% ---------------------------------------------------------------------------------

meta[~meta.name.isin(sources.name)]

# %% ---------------------------------------------------------------------------------

meta.to_(paths.outlet_meta)

# %% ---------------------------------------------------------------------------------
