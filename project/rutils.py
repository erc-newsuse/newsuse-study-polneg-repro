import pandas as pd
from brmspy.helpers import conversion
from rpy2 import robjects as ro

__all__ = ("df_to_r", "ro")


def df_to_r(data: pd.DataFrame) -> ro.DataFrame:
    """Convert a pandas DataFrame to an R DataFrame."""
    data = data.copy().convert_dtypes(dtype_backend="numpy_nullable")
    cat = {}
    for col in data.select_dtypes(["category"]):
        if pd.api.types.is_string_dtype(data[col].cat.categories):
            rtype = ro.StrVector
        elif pd.api.types.is_integer_dtype(data[col].cat.categories):
            rtype = ro.IntVector
        else:
            rtype = ro.FloatVector
        cat[col] = {
            "levels": rtype(data[col].cat.categories.tolist()),
            "ordered": data[col].cat.ordered,
        }
        data[col] = data[col].astype(data[col].cat.categories.dtype)

    rdf = conversion.py_to_r(data)
    for col, info in cat.items():
        idx = data.columns.tolist().index(col)
        rcol = ro.r["[["](rdf, col)
        rfac = ro.FactorVector(rcol, levels=info["levels"], ordered=info["ordered"])
        rdf[idx] = rfac  # type: ignore
    return rdf
