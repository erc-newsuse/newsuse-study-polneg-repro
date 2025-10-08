# %% ---------------------------------------------------------------------------------

import pandas as pd
from newsuse.data import DataFrame, strings
from tqdm.auto import tqdm

from project import paths

# %% Make posts ----------------------------------------------------------------------

missing = {"0": pd.NA, "": pd.NA}


def sanitize_strings(s: pd.Series) -> pd.Series:
    return pd.Series([t if pd.isnull(t) else strings.sanitize(t) for t in tqdm(s)])


posts = (
    pd.concat(
        {
            p.name.split(".")[0].split("-")[-1]: DataFrame.from_(p)
            for p in (paths.raw / "posts").glob("posts-*.parquet")
        }
    )
    .reset_index(level=0, names="country")
    .reset_index(drop=True)
    .assign(date=lambda df: pd.to_datetime(df["date"]))
    .assign(
        text=lambda df: sanitize_strings(df["text"]),
        link_title=lambda df: sanitize_strings(df["link_title"]),
        link_content=lambda df: sanitize_strings(df["link_content"]),
    )
    .replace({"link_title": missing, "link_content": missing})
    .rename(columns={"likes": "reactions"})
    .assign(
        reactions=lambda df: df["reactions"].astype(int),
        comments=lambda df: df["comments"].astype(int),
        shares=lambda df: df["shares"].astype(int),
    )
)

# %% ---------------------------------------------------------------------------------

posts.insert(0, "key", "sotrender@" + posts.pop("fb_post_id"))
del posts["hour"]
idx = posts.columns.tolist().index("text") + 1
for col in ["post_url", "author", "link_content", "link_title"]:
    posts.insert(idx, col, posts.pop(col))

# %% ---------------------------------------------------------------------------------

# %% ---------------------------------------------------------------------------------

text = posts[["key", "text", "link_title"]].rename(columns={"link_title": "title"})
mask = text[["text", "title"]].notnull().any(axis=1)
text = text[mask].reset_index(drop=True)
posts = posts.drop(columns=["text", "link_title"])[mask].reset_index(drop=True)

# %% ---------------------------------------------------------------------------------

posts.to_(paths.posts)
text.to_(paths.text)

# %% ---------------------------------------------------------------------------------
