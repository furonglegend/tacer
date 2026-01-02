"""
Code Accuracy Metrics
Evaluation metrics for DSL code generation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx

from dsl.executor import DSLExecutor, validate_script_syntax


def compute_execution_accuracy(generated_scripts: List[str], graphs: List[nx.Graph],
                                original_logits: List[int], 
                                executor: Optional[DSLExecutor] = None) -> Dict[str, float]:
    """
    Compute execution accuracy of generated DSL scripts.

    Args:
        generated_scripts: List of generated DSL code strings
        graphs: List of NetworkX graphs
        original_logits: List of original predictions
        executor: DSL executor instance

    Returns:
        Dict with accuracy metrics
    """
    if executor is None:
        executor = DSLExecutor(timeout=5.0)

    correct = 0
    executable = 0
    total = len(generated_scripts)
    errors = []
    absolute_errors = []

    for script, graph, target in zip(generated_scripts, graphs, original_logits):
        result, error = executor.execute(script, graph)

        if error is None:
            executable += 1

            # Check if result matches target
            if result is not None:
                if isinstance(result, (int, float)):
                    pred = int(result)
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    pred = int(result[0])
                else:
                    pred = -1

                if pred == target:
                    correct += 1

                absolute_errors.append(abs(pred - target))
        else:
            errors.append(error)

    accuracy = correct / total if total > 0 else 0
    execution_rate = executable / total if total > 0 else 0
    mean_ae = np.mean(absolute_errors) if absolute_errors else float('inf')

    return {
        'accuracy': accuracy,
        'execution_rate': execution_rate,
        'mean_absolute_error': mean_ae,
        'total_samples': total,
        'correct': correct,
        'executable': executable,
        'error_count': len(errors)
    }


def compute_script_validity(scripts: List[str], 
                            executor: Optional[DSLExecutor] = None) -> Dict[str, float]:
    """
    Check validity of generated scripts.

    Args:
        scripts: List of DSL scripts
        executor: DSL executor instance

    Returns:
        Dict with validity metrics
    """
    if executor is None:
        executor = DSLExecutor(timeout=5.0)

    syntactically_valid = 0
    semantically_valid = 0
    total = len(scripts)
    syntax_errors = []
    security_errors = []

    for script in scripts:
        # Check syntax
        is_valid, error = validate_script_syntax(script)
        if is_valid:
            syntactically_valid += 1

            # Check security (no forbidden patterns)
            try:
                executor.validate_script(script)
                semantically_valid += 1
            except Exception as e:
                security_errors.append(str(e))
        else:
            syntax_errors.append(error)

    return {
        'syntax_validity_rate': syntactically_valid / total if total > 0 else 0,
        'semantic_validity_rate': semantically_valid / total if total > 0 else 0,
        'total_scripts': total,
        'syntactically_valid': syntactically_valid,
        'semantically_valid': semantically_valid,
        'syntax_error_count': len(syntax_errors),
        'security_error_count': len(security_errors)
    }


def compute_script_complexity(scripts: List[str]) -> Dict[str, float]:
    """
    Analyze complexity of generated scripts.

    Args:
        scripts: List of DSL scripts

    Returns:
        Dict with complexity metrics
    """
    line_counts = []
    token_counts = []
    operation_counts = []

    dsl_operations = ['hop', 'filter', 'agg', 'select', 'classify', 
                      'count', 'intersect', 'union', 'difference']

    for script in scripts:
        lines = script.strip().split('\n')
        line_counts.append(len(lines))

        tokens = script.split()
        token_counts.append(len(tokens))

        # Count DSL operations
        op_count = sum(script.count(op) for op in dsl_operations)
        operation_counts.append(op_count)

    return {
        'mean_lines': np.mean(line_counts),
        'std_lines': np.std(line_counts),
        'mean_tokens': np.mean(token_counts),
        'std_tokens': np.std(token_counts),
        'mean_operations': np.mean(operation_counts),
        'std_operations': np.std(operation_counts)
    }


def compute_script_diversity(scripts: List[str]) -> Dict[str, float]:
    """
    Measure diversity of generated scripts.

    Args:
        scripts: List of DSL scripts

    Returns:
        Dict with diversity metrics
    """
    # Unique scripts
    unique_scripts = len(set(scripts))
    unique_ratio = unique_scripts / len(scripts) if scripts else 0

    # Unique first lines (different starting operations)
    first_lines = [s.split('\n')[0].strip() for s in scripts if s.strip()]
    unique_first = len(set(first_lines))

    # Operation distribution
    dsl_operations = ['hop', 'filter', 'agg', 'select', 'classify']
    op_counts = {op: 0 for op in dsl_operations}

    for script in scripts:
        for op in dsl_operations:
            if op in script:
                op_counts[op] += 1

    # Entropy of operation distribution
    total_ops = sum(op_counts.values())
    if total_ops > 0:
        probs = [c / total_ops for c in op_counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
    else:
        entropy = 0

    return {
        'unique_ratio': unique_ratio,
        'unique_scripts': unique_scripts,
        'unique_starts': unique_first,
        'operation_entropy': entropy,
        'operation_distribution': op_counts
    }
