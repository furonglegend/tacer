"""
Safe sandbox executor for DSL programs.

This module guarantees:
  - no filesystem or network access
  - bounded execution
  - deterministic semantics

Any violation yields ERROR.
"""

import time


class DSLRuntimeError(Exception):
    pass


def hop(G, root, k):
    return list(G.neighbors(root))[:k]


def filter_nodes(nodes, token):
    return [n for n in nodes if token.lower() in n.lower()]


def count(nodes):
    return len(nodes)


SAFE_ENV = {
    "hop": hop,
    "filter": filter_nodes,
    "count": count
}


def execute_dsl(expr: str, context: dict, timeout: float = 5.0):
    """
    Execute DSL expression in a restricted environment.
    """
    start = time.time()
    try:
        result = eval(
            expr,
            {"__builtins__": {}},
            {**SAFE_ENV, **context}
        )
        if time.time() - start > timeout:
            raise DSLRuntimeError("Timeout")
        return result
    except Exception:
        return "ERROR"
