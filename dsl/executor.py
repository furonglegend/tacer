"""
DSL Executor for Safe Script Execution
Provides sandboxed execution environment for DSL scripts.
"""

import ast
import signal
import threading
from typing import Dict, Any, Tuple, Optional
import networkx as nx

from .spec import DSL_OPERATIONS


class TimeoutError(Exception):
    """Exception raised when script execution times out."""
    pass


class DSLSecurityError(Exception):
    """Exception raised when script contains unsafe operations."""
    pass


class DSLExecutor:
    """
    Safe executor for DSL scripts with sandboxed environment.
    """

    def __init__(self, timeout: float = 5.0):
        """
        Initialize DSL executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.allowed_builtins = {
            'len': len,
            'range': range,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'max': max,
            'min': min,
            'sum': sum,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'True': True,
            'False': False,
            'None': None,
        }

        # Dangerous patterns to check for
        self.forbidden_patterns = [
            'import', 'exec', 'eval', 'compile', 'open', 'file',
            '__import__', '__builtins__', '__class__', '__bases__',
            '__subclasses__', '__globals__', '__code__', '__reduce__',
            'os.', 'sys.', 'subprocess', 'shutil', 'pickle',
        ]

    def validate_script(self, script: str) -> bool:
        """
        Validate script for security issues.

        Args:
            script: DSL script string

        Returns:
            True if script is safe, raises DSLSecurityError otherwise
        """
        # Check for forbidden patterns
        script_lower = script.lower()
        for pattern in self.forbidden_patterns:
            if pattern.lower() in script_lower:
                raise DSLSecurityError(f"Forbidden pattern detected: {pattern}")

        # Parse AST to check for unsafe constructs
        try:
            tree = ast.parse(script)
        except SyntaxError as e:
            raise DSLSecurityError(f"Syntax error in script: {e}")

        for node in ast.walk(tree):
            # Check for import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise DSLSecurityError("Import statements are not allowed")

            # Check for attribute access to dangerous objects
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    raise DSLSecurityError(f"Access to private attributes not allowed: {node.attr}")

        return True

    def execute(self, script: str, graph: nx.Graph, context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Optional[str]]:
        """
        Execute DSL script in sandboxed environment.

        Args:
            script: DSL script string
            graph: NetworkX graph to operate on
            context: Optional additional context variables

        Returns:
            Tuple of (result, error_message)
        """
        # Validate script first
        try:
            self.validate_script(script)
        except DSLSecurityError as e:
            return None, str(e)

        # Build execution environment
        env = {
            'G': graph,
            'graph': graph,
            **self.allowed_builtins,
            **DSL_OPERATIONS,
        }

        if context:
            env.update(context)

        # Execute with timeout
        result = [None]
        error = [None]

        def run_script():
            try:
                exec(script, {'__builtins__': self.allowed_builtins}, env)
                # Get result from environment if 'result' variable is set
                if 'result' in env:
                    result[0] = env['result']
                elif 'output' in env:
                    result[0] = env['output']
            except Exception as e:
                error[0] = str(e)

        thread = threading.Thread(target=run_script)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return None, f"Script execution timed out after {self.timeout} seconds"

        return result[0], error[0]

    def execute_and_get_logits(self, script: str, graph: nx.Graph, 
                                weights: Any = None) -> Tuple[Optional[int], Optional[str]]:
        """
        Execute script and get classification result.

        Args:
            script: DSL script that should produce a classification
            graph: NetworkX graph
            weights: Optional weight matrix for classification

        Returns:
            Tuple of (predicted_class, error_message)
        """
        context = {}
        if weights is not None:
            context['weights'] = weights

        result, error = self.execute(script, graph, context)

        if error:
            return None, error

        if isinstance(result, (int, float)):
            return int(result), None
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            return int(result[0]), None
        else:
            return None, "Script did not produce a valid classification result"


def validate_script_syntax(script: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a script is syntactically valid Python.

    Args:
        script: Script string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(script)
        return True, None
    except SyntaxError as e:
        return False, str(e)
