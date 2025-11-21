# %% ---------------------------------------------------------------------------------

import pandas as pd
from newspaper import Article
from newsuse.data import DataFrame
from newsuse.ml import pipeline
from tqdm.auto import tqdm

from project import paths

# %% ---------------------------------------------------------------------------------

sample = DataFrame.from_(paths.raw / "articles-sample.xlsx")

# %% ---------------------------------------------------------------------------------

# Modified article downloading with custom headers to avoid 406 errors
article_texts = []

# Define browser-like headers to prevent 406 Not Acceptable errors
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,"
    "application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Loop through all article URLs with progress bar
for url in tqdm(sample["link_url"]):
    if pd.isnull(url):
        article_texts.append(None)
        continue
    try:
        # Create article object with custom configuration
        article = Article(url)

        # Configure article with custom headers to mimic a browser
        article.config.browser_user_agent = headers["User-Agent"]
        article.config.headers = headers

        # Download and parse the article
        article.download()
        article.parse()
        content = f"{article.title.strip()}\n\n{article.text.strip()}"
        article_texts.append(content)
    except Exception as exc:
        # Log failures but continue processing
        article_texts.append(None)
        print(exc)

# %% ---------------------------------------------------------------------------------

sample["text"] = article_texts

# %% ---------------------------------------------------------------------------------

bad_prefixes = ["Yahoo is part of the Yahoo family of brands", "404", "Wyborcza.pl"]

sample = sample.dropna(subset=["text"])
for prefix in bad_prefixes:
    sample = sample[~sample["text"].str.startswith(prefix)]

# %% ---------------------------------------------------------------------------------

classifiers = {
    "political": pipeline("text-classification", paths.ml / "classifiers" / "political"),
    "valence": pipeline("text-classification", paths.ml / "classifiers" / "valence"),
}

for domain, model in classifiers.items():
    outputs = list(model(sample["text"], progress={"desc": domain}))
    sample[f"{domain}_text"] = [output["label"] for output in outputs]

# %% ---------------------------------------------------------------------------------

sample.to_(paths.articles_sample)

# %% ---------------------------------------------------------------------------------
