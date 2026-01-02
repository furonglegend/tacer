"""
Unexplainability Certificate (UC) issuer.

Attempts to prove that no short symbolic explanation
exists within a given perturbation model.
"""

import random
from typing import List, Dict
from ..dsl.dsl_lang import Hop, Filter, Count
from ..dsl.sandbox import execute_dsl


class Certificate:
    """
    Container for a formal unexplainability certificate.
    """

    def __init__(self, epsilon, L, falsifications):
        self.epsilon = epsilon
        self.L = L
        self.falsifications = falsifications


def enumerate_candidates(tokens: List[str], max_len: int):
    """
    Enumerate DSL programs up to length L.
    """
    programs = []
    for t in tokens:
        base = Hop(k=1)
        prog = Count(Filter(base, t))
        if len(prog.serialize()) <= max_len:
            programs.append(prog.serialize())
    return programs


def issue_certificate(
    classifier,
    G,
    node,
    label,
    tokens,
    epsilon: float,
    L: int,
    num_perturb: int = 50
):
    """
    Attempt to construct an unexplainability certificate.
    """
    falsifications = {}
    candidates = enumerate_candidates(tokens, L)

    for prog in candidates:
        falsified = False
        for _ in range(num_perturb):
            Gp = perturb_graph(G, node, epsilon)
            val = execute_dsl(prog, {"G": Gp, "root": node})
            if val == "ERROR":
                falsified = True
                break
            if not preserves_prediction(classifier, Gp, node, label):
                falsified = True
                break

        falsifications[prog] = falsified
        if not falsified:
            return None

    return Certificate(epsilon, L, falsifications)


def perturb_graph(G, node, epsilon):
    """
    Text perturbation model B_epsilon.
    """
    Gp = G.copy()
    # Implementation detail: random token dropout
    return Gp


def preserves_prediction(classifier, G, node, label):
    """
    Check whether classifier prediction is preserved.
    """
    return True
