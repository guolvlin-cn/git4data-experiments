"""Real tools the agent can call. Plain Python functions — they actually run."""
import ast
import operator

# small local knowledge base (a real kb_lookup, raises on unknown -> error span)
_KB = {
    "france population": 67_000_000,
    "japan population": 125_000_000,
    "speed of light": 299_792,          # km/s
    "hamlet": "William Shakespeare",
    "the odyssey": "Homer",
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str):
    """Evaluate an arithmetic expression safely (no eval())."""
    result = _safe_eval(ast.parse(expression, mode="eval"))
    return round(result, 4) if isinstance(result, float) else result


def kb_lookup(entity: str):
    """Look up a fact; raises KeyError on unknown entity (-> tool error span)."""
    key = entity.strip().lower()
    if key not in _KB:
        raise KeyError(f"no fact for '{entity}'")
    return _KB[key]


TOOLS = {
    "calculator": calculator,
    "kb_lookup": kb_lookup,
}

# OpenAI/Anthropic-style tool schemas (used when a real LLM backend is active)
TOOL_SCHEMAS = [
    {"name": "calculator", "description": "Evaluate an arithmetic expression, e.g. '67000000 * 2'.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"]}},
    {"name": "kb_lookup", "description": "Look up a fact about an entity (population, author, constants).",
     "parameters": {"type": "object", "properties": {"entity": {"type": "string"}},
                    "required": ["entity"]}},
]
