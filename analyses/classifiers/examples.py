# %% ---------------------------------------------------------------------------------

import numpy as np
from newsuse.data import DataFrame

from project import paths

rng = np.random.default_rng(17)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.final)
    .merge(DataFrame.from_(paths.text, columns=["key", "fulltext"]))
    .rename(columns={"fulltext": "text"})
)

# %% ---------------------------------------------------------------------------------

examples = (
    data.query("country == 'us'")
    .groupby(["political", "valence"])
    .sample(1, random_state=rng)[["key", "political", "event", "sentiment", "text"]]
)

# %% ---------------------------------------------------------------------------------

print(
    examples.drop(columns=["key"])
    .set_index(["political", "event", "sentiment"])
    .rename({0: "Non-political", 1: "Political"}, axis=0, level="political")
    .style.format(escape="latex")
    .to_latex(hrules=True, multirow_align="t")
)

# %% ---------------------------------------------------------------------------------
