import json
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from rpy2 import robjects as ro

__all__ = (
    "index_idata",
    "rebuild_model",
    "store_model_metadata",
    "hdi",
    "eti",
    "contr_effect",
    "contr_ref",
)

DataCoordsT = Mapping[str, np.ndarray | tuple[str, np.ndarray]]
FormulaT = ro.Formula | str
OBS_DIM = "__obs__"
GROUP_DIM = "__group__"


def store_model_metadata(
    idata: az.InferenceData,
    model: bmb.Model,
    *,
    formula: str | None = None,
    family: str | None = None,
    response: str | None = None,
) -> az.InferenceData:
    """Store Bambi model metadata in InferenceData attributes.

    This function stores metadata needed to reconstruct a Bambi model
    from an ArviZ InferenceData object. It is the counterpart to
    :func:`rebuild_model`.

    Parameters
    ----------
    idata
        ArviZ InferenceData object to store metadata in.
    model
        Fitted Bambi model.
    formula
        Model formula string. If None, extracted from `model.formula`.
    family
        Model family name. If None, extracted from `model.family.name`.
    response
        Response variable name.
        If None, extracted from `model.response_component.response_term.name`.

    Returns
    -------
    az.InferenceData
        The same InferenceData object with metadata attributes added.

    Notes
    -----
    The following attributes are stored:
    - ``categorical_specs``: JSON-encoded dict of categorical column specifications
    - ``formula``: Model formula string
    - ``family``: Model family name
    - ``target``: Response variable name
    """
    # Extract categorical specifications from model data
    cat_specs = {
        col: {
            "categories": list(model.data[col].cat.categories),
            "ordered": model.data[col].cat.ordered,
        }
        for col in model.data.select_dtypes("category").columns
    }
    idata.attrs["categorical_specs"] = json.dumps(cat_specs)

    # Store formula (clean up whitespace)
    if formula is None:
        formula = str(model.formula)
    idata.attrs["formula"] = formula.strip().replace("\n", " ")

    # Store family
    if family is None:
        family = model.family.name
    idata.attrs["family"] = family

    # Store target variable name
    if response is None:
        response = model.response_component.response_term.name
    idata.attrs["response"] = response

    return idata


def rebuild_model(idata: az.InferenceData) -> bmb.Model:
    """Rebuild Bambi model from InferenceData attributes.

    This function reconstructs a Bambi model from metadata stored in
    an ArviZ InferenceData object. The InferenceData must contain
    the following attributes: 'formula', 'family', and 'categorical_specs'.

    Parameters
    ----------
    idata
        ArviZ InferenceData object with model metadata in attributes.
        Must have 'formula', 'family', and 'categorical_specs' attributes.

    Returns
    -------
    bmb.Model
        Reconstructed Bambi model (not fitted).

    Raises
    ------
    KeyError
        If required attributes are missing from idata.
    """
    cat_specs = json.loads(idata.attrs["categorical_specs"])
    model_data = idata.observed_data.to_dataframe().reset_index()
    for col, spec in cat_specs.items():
        model_data[col] = pd.Categorical(
            model_data[col],
            categories=spec["categories"],
            ordered=spec["ordered"],
        )

    # Handle compound formulas (e.g., hurdle models with `;` separators)
    formula_str = idata.attrs["formula"]
    if ";" in formula_str:
        formula_parts = [part.strip() for part in formula_str.split(";")]
        formula = bmb.Formula(*formula_parts)
    else:
        formula = formula_str

    return bmb.Model(
        formula,
        model_data,
        family=idata.attrs["family"],
    )


def hdi(s: pd.Series, prob: float | None = None) -> pd.Series:
    """Compute HDI for a pandas Series."""
    if prob is None:
        prob = az.rcParams["stats.ci_prob"]
    hdi_bounds = az.hdi(s.values, hdi_prob=prob)
    return pd.Series({"median": s.median(), "lower": hdi_bounds[0], "upper": hdi_bounds[1]})


def eti(
    s: pd.Series,
    *,
    alpha: float | None = None,
    method: str = "inverted_cdf",
    **kwargs: Any,
) -> pd.Series:
    """Compute Equal-Tailed Interval (ETI) for a pandas Series."""
    if alpha is None:
        alpha = float(1 - az.rcParams["stats.ci_prob"])
    return pd.Series(
        np.quantile(s, [0.5, alpha / 2, 1 - alpha / 2], method=method, **kwargs),
        index=["median", "lower", "upper"],
    )


def index_idata(idata: az.InferenceData, xindex: Sequence[Hashable]) -> az.InferenceData:
    """Set xindex for all groups in InferenceData.

    The xindex is used by ArviZ for plotting and indexing. The function is a convenience
    wrapper around xarray's `set_xindex` method.

    Parameters
    ----------
    idata
        An ArviZ InferenceData object.
    xindex
        A sequence of coordinate names to set as xindex.
    """
    for group in idata.groups():
        ds = getattr(idata, group)
        xs = [col for col in xindex if col in ds.coords]
        if not xs:
            continue
        ds = ds.set_xindex(xs)
        setattr(idata, group, ds)
    return idata


def contr_effect(s: pd.Series, level: int | str = 0) -> pd.Series:
    """Effect coding contrasts."""
    index = s.index.get_level_values(level)
    x = s.to_numpy()
    contr = x - (x.sum() - x) / (x.size - 1)
    contr = pd.Series(contr, index=pd.Series(index, name="contrast"))
    return contr


def contr_ref(s: pd.Series, ref: int | str, level: int | str) -> pd.Series:
    """Reference category contrasts."""
    index = s.index.get_level_values(level)
    x = s.to_numpy()
    mask = index == ref
    if not mask.any():
        index = pd.Series([], dtype=index.dtype, name="contrast")
        return pd.Series([], dtype=s.dtype, index=index)
    contr = x[~mask] - x[mask]
    return pd.Series(contr, index=pd.Series(index[~mask], name="contrast"))
