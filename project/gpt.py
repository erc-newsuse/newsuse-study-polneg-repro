from typing import Literal

from pydantic import BaseModel

__all__ = ("NegativityClassification",)

Score = Literal[-1, 0, 1]


class NegativityClassification(BaseModel):
    event: Score
    sentiment: Score
