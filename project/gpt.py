from typing import Literal

from pydantic import BaseModel

__all__ = ("NegativityClassification",)

Score = Literal["negative", "neutral", "positive"]


class NegativityClassification(BaseModel):
    event: Score
    sentiment: Score
