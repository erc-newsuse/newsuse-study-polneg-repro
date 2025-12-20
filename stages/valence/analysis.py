# %% ---------------------------------------------------------------------------------
"""Analysis of political valence differences using arviz/bambi machinery.

This script tests whether the properly marginalized expected class probabilities
differ between political and non-political posts overall and by country.
Results are illustrated by point+interval estimates for class probabilities
and corresponding odds ratios.
"""

import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
import xarray as xr
from scipy.special import logit

from project import config, paths
from project.bayes import index_idata, rebuild_model

xr.set_options(**config.xarray)
az.rcParams.update(config.arviz)
mpl.rcParams.update(config.plotting.params)
so.Plot.config.theme.update(config.plotting.params)

# Configuration
target = os.environ.get("TARGET")
if target is None:
    target = input("Enter target (event): ").strip() or "event"
support = [*config.categorical[target]]
opts = config.glmm.valence.targets[target]

figpath = paths.figures / "valence" / target
figpath.mkdir(parents=True, exist_ok=True)

countries_map = config.categorical.country
political_map = dict(enumerate(config.categorical.political))

# %% Load inference data -------------------------------------------------------------

idata = az.from_netcdf(paths.glmm / "valence" / f"{target}.nc")
idata = index_idata(idata, ["key", *sum(opts.predictors.values(), start=[])])

# Rebuild model for bambi.interpret functions
model = rebuild_model(idata)

# Extract posterior expected probabilities
epred = az.extract(idata, group="posterior_epred")
# Marginalize out weekend effects to focus on main effects
if "weekend" in epred.coords:
    # Average over weekend levels while preserving other coordinates
    weekend_vals = np.unique(epred.coords["weekend"].values)
    epred = sum(epred.sel(weekend=w).drop_vars("weekend") for w in weekend_vals) / len(
        weekend_vals
    )

# Get probabilities as xarray DataArray
probs = epred.p

# %% Compute posterior expectations using az.hdi ------------------------------------


def summarize_probabilities(
    probs: xr.DataArray,
    by: list[str] | None = None,
    prob: float = az.rcParams["stats.ci_prob"],
) -> pd.DataFrame:
    """Summarize posterior probabilities with HDI.

    Parameters
    ----------
    probs
        Probability array with dimensions including sample and target.
    by
        Grouping dimensions for the summary.
    prob
        HDI probability.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with median and HDI bounds.
    """
    if by:
        grouped = probs.groupby(by)
        results = []
        for key, group in grouped:
            median_val = group.median("sample").values
            hdi_vals = az.hdi(group.values.flatten(), hdi_prob=prob)
            if isinstance(key, tuple):
                row = dict(zip(by, key, strict=True))
            else:
                row = {by[0]: key}
            row["median"] = float(median_val) if np.ndim(median_val) == 0 else median_val
            row["lower"] = hdi_vals[0]
            row["upper"] = hdi_vals[1]
            results.append(row)
        return pd.DataFrame(results)
    median = float(probs.median("sample").values)
    hdi = az.hdi(probs.values.flatten(), hdi_prob=prob)
    return pd.DataFrame([{"median": median, "lower": hdi[0], "upper": hdi[1]}])


def compute_probability_summary(
    probs: xr.DataArray,
    target_name: str,
    prob: float = az.rcParams["stats.ci_prob"],
) -> pd.DataFrame:
    """Compute probability summary by political status and country.

    Parameters
    ----------
    probs
        Posterior probability samples.
    target_name
        Name of the target variable.
    prob
        HDI probability mass.

    Returns
    -------
    pd.DataFrame
        Combined overall and by-country summaries.
    """
    # Convert to DataFrame for easier manipulation
    df = probs.to_dataframe(name="prob").reset_index()

    # Overall (average across countries within each sample)
    overall_samples = (
        df.groupby(["political", "chain", "draw", target_name])["prob"].mean().reset_index()
    )
    overall = (
        overall_samples.groupby(["political", target_name])
        .apply(
            lambda g: pd.Series(
                {
                    "median": g["prob"].median(),
                    "lower": az.hdi(g["prob"].values, hdi_prob=prob)[0],
                    "upper": az.hdi(g["prob"].values, hdi_prob=prob)[1],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    overall.insert(0, "country", "overall")

    # By country
    by_country = (
        df.groupby(["country", "political", target_name])
        .apply(
            lambda g: pd.Series(
                {
                    "median": g["prob"].median(),
                    "lower": az.hdi(g["prob"].values, hdi_prob=prob)[0],
                    "upper": az.hdi(g["prob"].values, hdi_prob=prob)[1],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return pd.concat([overall, by_country], ignore_index=True)


# Compute summaries
prob_summary = compute_probability_summary(probs, target)


# %% Plot posterior expectations -----------------------------------------------------

country_order = ["overall", *config.categorical.country]
fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = prob_summary.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=target, y="median", color="political")
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.political])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else countries_map[country]
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.set_xticks(support)

fig.legends.clear()
axes[0].set_ylabel("Posterior expectation", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "posterior-expectations.pdf")


# %% Compute odds ratios using arviz HDI --------------------------------------------


def compute_odds_ratio_summary(
    probs: xr.DataArray,
    target_name: str,
    prob: float = az.rcParams["stats.ci_prob"],
) -> pd.DataFrame:
    """Compute odds ratio summary (political vs non-political) by country.

    Parameters
    ----------
    probs
        Posterior probability samples.
    target_name
        Name of the target variable.
    prob
        HDI probability mass.

    Returns
    -------
    pd.DataFrame
        Odds ratio summaries with HDI.
    """
    df = probs.to_dataframe(name="prob").reset_index()

    # Compute log-odds for each sample, then difference
    df["logit_prob"] = logit(df["prob"].clip(1e-10, 1 - 1e-10))

    # Pivot to get political and non-political side by side
    pivot = df.pivot_table(
        index=["country", "chain", "draw", target_name],
        columns="political",
        values="logit_prob",
    ).reset_index()

    # Odds ratio: exp(logit(p_political) - logit(p_nonpolitical))
    # political=1, non-political=0
    pivot["log_or"] = pivot[1] - pivot[0]
    pivot["or"] = np.exp(pivot["log_or"])

    # Overall (average log-OR across countries within each sample)
    overall_samples = (
        pivot.groupby(["chain", "draw", target_name])["log_or"].mean().reset_index()
    )
    overall_samples["or"] = np.exp(overall_samples["log_or"])
    overall = (
        overall_samples.groupby(target_name)
        .apply(
            lambda g: pd.Series(
                {
                    "median": np.exp(g["log_or"].median()),
                    "lower": np.exp(az.hdi(g["log_or"].values, hdi_prob=prob)[0]),
                    "upper": np.exp(az.hdi(g["log_or"].values, hdi_prob=prob)[1]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    overall.insert(0, "country", "overall")

    # By country
    by_country = (
        pivot.groupby(["country", target_name])
        .apply(
            lambda g: pd.Series(
                {
                    "median": np.exp(g["log_or"].median()),
                    "lower": np.exp(az.hdi(g["log_or"].values, hdi_prob=prob)[0]),
                    "upper": np.exp(az.hdi(g["log_or"].values, hdi_prob=prob)[1]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return pd.concat([overall, by_country], ignore_index=True)


# Compute odds ratio summaries
or_summary = compute_odds_ratio_summary(probs, target)


# %% Plot odds ratios ----------------------------------------------------------------

fig, axes = plt.subplots(ncols=len(country_order), figsize=(21, 3), sharey=True)

for ax, country in zip(axes, country_order, strict=True):
    gdf = or_summary.query("country == @country")
    range_kw = config.plotting.objects.range
    (
        so.Plot(gdf, x=target, y="median", color=target)
        .add(so.Dot(**config.plotting.objects.dot), so.Dodge())
        .add(so.Range(**range_kw), so.Dodge(), ymin="lower", ymax="upper")
        .scale(color=[*config.plotting.color.valence])
        .on(ax)
        .plot()
    )
    title = country.title() if country == "overall" else countries_map[country]
    ax.set_title(title, fontsize="xx-large")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_yscale("log")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(10**-1, 10**1)
    ax.set_xticks(support)

fig.legends.clear()
axes[0].set_ylabel("Posterior odds ratio", fontsize="xx-large")
fig.tight_layout()
fig.savefig(figpath / "posterior-odds-ratio.pdf")


# %% Summary statistics using az.summary --------------------------------------------

print("\n" + "=" * 60)
print("Probability Summary (median [HDI])")
print("=" * 60)
print(prob_summary.to_string(index=False))

print("\n" + "=" * 60)
print("Odds Ratio Summary (political vs non-political)")
print("=" * 60)
print(or_summary.to_string(index=False))


# %% ---------------------------------------------------------------------------------
