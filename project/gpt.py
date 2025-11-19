from typing import Literal

from pydantic import BaseModel

__all__ = ("ValenceClassification", "EventClassification", "SentimentClassification")

Score = Literal[-1, 0, 1]
# Score = Literal["NEGATIVE", "NEUTRAL", "POSITIVE"]


class EventClassification(BaseModel):
    event: Score


class SentimentClassification(BaseModel):
    sentiment: Score


class ValenceClassification(BaseModel):
    event: Score
    sentiment: Score
