"""DSL module for executable graph explanations."""

from .spec import (
    hop, filter_nodes, agg, select, classify,
    get_node_text, get_node_label, count_nodes,
    intersect, union, difference, DSL_OPERATIONS
)
from .executor import DSLExecutor, validate_script_syntax

__all__ = [
    'hop', 'filter_nodes', 'agg', 'select', 'classify',
    'get_node_text', 'get_node_label', 'count_nodes',
    'intersect', 'union', 'difference', 'DSL_OPERATIONS',
    'DSLExecutor', 'validate_script_syntax'
]
