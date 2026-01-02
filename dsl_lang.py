"""
Domain-Specific Language (DSL) for executable explanations.

This DSL is intentionally minimal, symbolic, and enumerable.
It enables formal falsification and certificate construction.
"""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class DSLExpr:
    """
    Base class for all DSL expressions.
    """
    def serialize(self) -> str:
        raise NotImplementedError


@dataclass
class Hop(DSLExpr):
    """
    hop(G, root, k)
    """
    k: int

    def serialize(self) -> str:
        return f"hop(G, root, {self.k})"


@dataclass
class Filter(DSLExpr):
    """
    filter(nodes, token)
    """
    source: DSLExpr
    token: str

    def serialize(self) -> str:
        return f"filter({self.source.serialize()}, '{self.token}')"


@dataclass
class Count(DSLExpr):
    """
    count(nodes)
    """
    source: DSLExpr

    def serialize(self) -> str:
        return f"count({self.source.serialize()})"
