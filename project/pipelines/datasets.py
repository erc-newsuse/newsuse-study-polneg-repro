from collections.abc import Iterable, Sequence
from functools import singledispatchmethod
from typing import Any

import pandas as pd
import torch
import torch.utils
import torch.utils.data
from transformers.pipelines.pt_utils import KeyDataset as _KeyDataset

__all__ = (
    "SimpleDataset",
    "KeyDataset",
)


_ExampleT = dict[str, Any]


class SimpleDataset(torch.utils.data.Dataset):
    """Simple indexable :mod:`torch` dataset
    fetching examples from elements of ``self.data``.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1,2,3], "b": [1,2,3]})
    >>> dataset = SimpleDataset(df)
    >>> dataset[0]
    {'a': 1, 'b': 1}
    >>> dataset[:2]
    {'a': [1, 2], 'b': [1, 2]}

    It is compatible with :class:`newsuse.ml.KeyDaset`.

    >>> keyed = KeyDataset(dataset, "a")
    >>> keyed[0]
    1
    >>> keyed[:2]
    [1, 2]
    """

    def __init__(
        self, data: Sequence[_ExampleT] | Iterable[_ExampleT] | pd.DataFrame
    ) -> None:
        if not isinstance(data, Sequence | pd.DataFrame):
            data = list(data)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> _ExampleT:
        return self._get(self.data, idx)

    @singledispatchmethod
    def _get(self, data, idx: int) -> _ExampleT:
        return data[idx]

    @_get.register
    def _(self, data: pd.Series, idx: int | slice) -> _ExampleT:
        out = data.iloc[idx]
        if isinstance(idx, slice):
            out = out.tolist()
        return out

    @_get.register
    def _(self, data: pd.DataFrame, idx: int | slice) -> _ExampleT:
        out = data.iloc[idx]
        if isinstance(idx, slice):
            return out.to_dict(orient="list")
        return out.to_dict()


class KeyDataset(_KeyDataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset | Sequence | pd.DataFrame, key: str
    ) -> None:
        if not isinstance(dataset, torch.utils.data.Dataset):
            dataset = SimpleDataset(dataset)
        super().__init__(dataset, key)
