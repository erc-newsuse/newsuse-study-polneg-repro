from collections.abc import Iterator, Mapping
from typing import Any

import arviz as az
import matplotlib as mpl
import numpy as np

__all__ = ("ArvizLabeller", "make_legend", "annotate_ci")


def annotate_ci(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    bounds: tuple[float, float],
    *,
    alpha: float = 0.05,
    digits: int = 3,
    template: str = r"{conf:.0f}\% CI: [{lb:.{digits}f}, {ub:.{digits}f}]",
    prefix: str = "",
    suffix: str = "",
    offset: float = 0.05,
    marker: str = "*",
    size: int = 200,
    color: str = "red",
    edgecolor: str = "black",
    marker_kwargs: Mapping[str, Any] | None = None,
    marker_offset: float | None = None,
    bbox: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> mpl.axes.Axes:
    lb, ub = bounds
    x, y = xy
    conf = 100 * (1 - alpha)
    text = template.format(conf=conf, lb=lb, ub=ub, digits=digits)
    text = f"{prefix}{text}{suffix}"
    if np.unique(np.sign(bounds)).size == 1:
        if marker_offset is None:
            marker_offset = offset * 2
        ax.scatter(
            x,
            y + marker_offset,
            s=size,
            marker=marker,
            color=color,
            edgecolor=edgecolor,
            zorder=10,
            **(marker_kwargs or {}),
        )
    kwargs = {
        "ha": "center",
        "va": "center",
        "fontsize": 6,
        "xycoords": "data",
        "textcoords": "offset points",
        "bbox": {"fc": "white", **(bbox or {})},
        **kwargs,
    }
    ax.annotate(text, xy=(x, y + offset), xytext=(0, 5), **kwargs)
    return ax


def make_legend(
    fig: mpl.figure.Figure | mpl.axes.Axes,
    bbox_to_anchor: tuple[float, float] | None = None,
    loc: str | None = "upper right",
    *,
    title: str | None = None,
    visible: bool = False,
) -> mpl.legend.Legend:
    legend = fig.legends[0] if isinstance(fig, mpl.figure.Figure) else fig.get_legend()
    if bbox_to_anchor:
        legend.set_bbox_to_anchor(bbox_to_anchor)
    if loc:
        legend.set_loc(loc)
    legend.set_title(title)
    legend.get_frame().set_visible(visible)
    return legend


class ArvizLabeller(az.labels.BaseLabeller):
    def make_label_vert(self, var_name, *_) -> str:
        kind, var_name = var_name.split("_", 1)
        term = " * ".join(self._iter_parts(var_name))
        return f"[{kind}] {term}"

    def _iter_parts(self, var_name: str) -> Iterator[str]:
        for pat in ("__", ":"):
            if pat in var_name:
                for part in var_name.split(pat):
                    yield from self._iter_parts(part)
                return
        else:
            yield self._process_var_name(var_name)

    def _process_var_name(self, var_name: str) -> str:
        if var_name.startswith(prefix := "country"):
            return var_name.removeprefix(prefix).upper().strip()
        if var_name.startswith(prefix := "political"):
            return var_name.removeprefix(prefix).upper().strip()
        if var_name.startswith(prefix := "Intercept"):
            return "Threshold" + var_name.removeprefix(prefix)
        return var_name
