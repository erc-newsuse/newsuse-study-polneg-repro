# %% ---------------------------------------------------------------------------------

import numpy as np
from newsuse.data import DataFrame

from project import paths

rng = np.random.default_rng(303)

# %% ---------------------------------------------------------------------------------

data = (
    DataFrame.from_(paths.final)
    .merge(DataFrame.from_(paths.text, columns=["key", "fulltext"]))
    .rename(columns={"fulltext": "text"})
)

# %% ---------------------------------------------------------------------------------

keys_non_political = [
    "sotrender@15704546335_10162552080806336",
    "sotrender@95475020353_709522191045874",
    "sotrender@15704546335_10159559205006336",
    "sotrender@210277954204_10158381607114205",
    "sotrender@5550296508_10162702461066509",
    "sotrender@21898300328_10162555563490329",
    "sotrender@18343191100_10159272109141101",
    "sotrender@182919686769_10158647635521770",
]

keys_political = [
    "sotrender@18468761129_10158064770286130",
    "sotrender@7533944086_641857587802666",
    "sotrender@18343191100_10159611106021101",
    "sotrender@182919686769_10158414253161770",
    "sotrender@7533944086_722868099701614",
    "sotrender@19013582168_10157646036487169",
    "sotrender@338028696036_447511823901175",
    "sotrender@85452072376_10160035542277377",
]

# %% ---------------------------------------------------------------------------------

examples_main_text = (
    data[data["key"].isin(keys_non_political + keys_political)]
    .groupby(["event", "sentiment"])
    .sample(1, random_state=rng)[
        ["key", "political", "event", "sentiment", "valence", "text"]
    ]
)

# %% ---------------------------------------------------------------------------------

print(
    examples_main_text.drop(columns=["key"])
    .set_index(["political", "event", "sentiment"])
    .sort_index()
    .reset_index("sentiment")
    .rename({0: "Non-political", 1: "Political"}, axis=0, level="political")
    .style.format(escape="latex")
    .to_latex(hrules=True, multirow_align="t")
)

# %% ---------------------------------------------------------------------------------
