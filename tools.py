"""Tool implementations. Each tool takes a single string argument."""

import ast
import math
import operator
from datetime import datetime, timezone

# --- calculator: safe arithmetic via AST (no eval) ---
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {name: getattr(math, name) for name in
              ["sqrt", "sin", "cos", "tan", "log", "log10", "exp", "ceil", "floor", "fabs"]}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. calculator[(120/60)] or calculator[sqrt(81)]."""
    try:
        return str(_eval(ast.parse(expression.strip(), mode="eval")))
    except Exception as e:
        return f"Error: {e}"


_LENGTH_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}
_TIME_TO_S = {"ms": 0.001, "s": 1.0, "sec": 1.0, "seconds": 1.0, "min": 60.0,
              "minutes": 60.0, "h": 3600.0, "hr": 3600.0, "hours": 3600.0}


def unit_convert(argument: str) -> str:
    """Convert units, e.g. unit_convert[2.5,km,mi] or unit_convert[120,minutes,seconds]."""
    parts = [p.strip() for p in argument.split(",")]
    if len(parts) != 3:
        return "Error: expected format value,from_unit,to_unit — e.g. 2.5,km,mi"
    try:
        value = float(parts[0])
    except ValueError:
        return f"Error: '{parts[0]}' is not a number"
    from_u, to_u = parts[1], parts[2]
    for table in (_LENGTH_TO_M, _TIME_TO_S):
        if from_u in table and to_u in table:
            return str(value * table[from_u] / table[to_u])
    return f"Error: unsupported units '{from_u}' -> '{to_u}'"


def get_time(argument: str) -> str:
    """Current UTC time; argument is an optional label, e.g. get_time[trip start]."""
    label = argument.strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{label}: {now}" if label else now


def word_count(argument: str) -> str:
    """Count words in the given text, e.g. word_count[hello brave new world]."""
    return str(len(argument.split()))


TOOLS = {
    "calculator": calculator,
    "unit_convert": unit_convert,
    "get_time": get_time,
    "word_count": word_count,
}
