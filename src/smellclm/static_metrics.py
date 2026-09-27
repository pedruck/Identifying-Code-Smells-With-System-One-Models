"""Deterministic method-level metrics (evidence for reviewers and the rule baseline).

Supported: Python (``ast``, with a lexical fallback) and the brace languages
Java, JavaScript, TypeScript, C++, C#, Kotlin and Go (a comment/string-aware
lexer).  The metrics are deliberately simple and fully specified so that the
rule baseline is reproducible; they are not a replacement for a parser-based
extractor.

* ``effective_loc``    non-blank lines that contain code after removing comments
* ``parameter_count``  declared parameters of the first signature, excluding the
                       receiver (``self``/``cls`` in Python, ``this`` in TypeScript)
                       and C/C++ ``(void)``
* ``max_nesting``      deepest nesting of control-flow blocks inside the body
                       (the body itself is depth 0; ``else``/``elif``/``catch``
                       branches are siblings, not deeper levels)
"""
from __future__ import annotations

import ast
import re
import textwrap

METRICS_VERSION = "static-metrics/1"

BRACE_LANGUAGES = {"java", "javascript", "typescript", "cpp", "csharp", "kotlin", "go"}
CONTROL_KEYWORDS = {"if", "else", "for", "foreach", "while", "do", "switch", "try", "catch", "finally",
                    "when", "select", "using", "lock", "with"}
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def compute(code: str, language: str) -> dict:
    if language == "python":
        return _python(code)
    if language in BRACE_LANGUAGES:
        return _brace(code, language)
    raise ValueError(f"no static metrics for language {language!r}")


# --------------------------------------------------------------------------- brace languages
def strip_comments_and_strings(code: str, language: str) -> str:
    """Replace comments with nothing and string/char literals with ``""``, keeping newlines."""
    out, i, n = [], 0, len(code)
    quotes = "\"'`" if language in {"javascript", "typescript", "go", "kotlin"} else "\"'"
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = code.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and nxt == "*":
            j = code.find("*/", i + 2)
            seg = code[i:n if j < 0 else j + 2]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        elif c in quotes:
            j, lines = i + 1, 0
            while j < n and code[j] != c:
                if code[j] == "\\":
                    j += 1
                elif code[j] == "\n":
                    lines += 1
                    if c != "`" and language != "kotlin":
                        break
                j += 1
            out.append('""' + "\n" * lines)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _split_top_level(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _signature_params(stripped: str, language: str) -> int:
    # the first balanced parenthesis group that precedes the body (or an arrow)
    start = stripped.find("(")
    if start < 0:
        return 0
    depth = 0
    for j in range(start, len(stripped)):
        if stripped[j] == "(":
            depth += 1
        elif stripped[j] == ")":
            depth -= 1
            if depth == 0:
                inner = stripped[start + 1:j]
                break
    else:
        return 0
    # Go methods declare the receiver in the first group: func (r *T) Name(a int)
    if language == "go" and re.match(r"\s*func\s*\(", stripped):
        rest = stripped[j + 1:]
        m = re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*\(", rest)
        if m:
            return _signature_params(rest, "go-plain")
    params = _split_top_level(inner)
    if language in {"cpp", "go-plain"} and params == ["void"]:
        return 0
    if language == "typescript" and params and re.match(r"this\s*:", params[0]):
        params = params[1:]
    return len(params)


def _brace_nesting(stripped: str) -> int:
    """Max depth of control-flow braces nested inside the outermost body brace."""
    stack: list[bool] = []          # True for a control-flow block
    header: list[str] = []          # identifiers seen since the last statement boundary
    best, parens = 0, 0
    for tok in re.finditer(r"[A-Za-z_$][A-Za-z0-9_$]*|[{};()]", stripped):
        t = tok.group()
        if t in "()":
            parens += 1 if t == "(" else -1
        elif t == ";" and parens > 0:
            continue                # for (init; cond; step) is still the loop header
        elif t == "{":
            is_control = bool(stack) and any(w in CONTROL_KEYWORDS for w in header)
            stack.append(is_control)
            best = max(best, sum(stack))
            header = []
        elif t == "}":
            if stack:
                stack.pop()
            header = []
        elif t == ";":
            header = []
        else:
            header.append(t)
    return best


def _brace(code: str, language: str) -> dict:
    stripped = strip_comments_and_strings(code, language)
    loc = sum(1 for line in stripped.splitlines() if line.strip())
    return {"metrics_version": METRICS_VERSION, "effective_loc": loc,
            "parameter_count": _signature_params(stripped, language),
            "max_nesting": _brace_nesting(stripped)}


# --------------------------------------------------------------------------- python
_PY_BLOCKS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
if hasattr(ast, "Match"):
    _PY_BLOCKS = _PY_BLOCKS + (ast.Match,)
if hasattr(ast, "TryStar"):
    _PY_BLOCKS = _PY_BLOCKS + (ast.TryStar,)


def _py_depth(nodes, depth: int) -> int:
    best = depth
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If):
            best = max(best, _py_depth(node.body, depth + 1))
            orelse = node.orelse
            # elif chains are siblings of the original if
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                best = max(best, _py_depth(orelse[0].body, depth + 1))
                orelse = orelse[0].orelse
            best = max(best, _py_depth(orelse, depth + 1))
        elif isinstance(node, _PY_BLOCKS):
            for fld in ("body", "orelse", "finalbody"):
                best = max(best, _py_depth(getattr(node, fld, []) or [], depth + 1))
            for h in getattr(node, "handlers", []) or []:
                best = max(best, _py_depth(h.body, depth + 1))
            for c in getattr(node, "cases", []) or []:
                best = max(best, _py_depth(c.body, depth + 1))
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    best = max(best, _py_depth([child], depth))
    return best


def _python(code: str) -> dict:
    src = textwrap.dedent(code)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _python_lexical(src)
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    comment_free = _python_code_lines(src)
    if fn is None:
        return {"metrics_version": METRICS_VERSION, "effective_loc": len(comment_free),
                "parameter_count": 0, "max_nesting": _py_depth(tree.body, 0), "parser": "ast"}
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    names = [p.arg for p in params] + ([a.vararg.arg] if a.vararg else []) + ([a.kwarg.arg] if a.kwarg else [])
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    return {"metrics_version": METRICS_VERSION, "effective_loc": len(comment_free),
            "parameter_count": len(names), "max_nesting": _py_depth(fn.body, 0), "parser": "ast"}


def _python_code_lines(src: str) -> list[str]:
    import io
    import tokenize
    lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            for ln in range(tok.start[0], tok.end[0] + 1):
                lines.add(ln)
    except (tokenize.TokenError, IndentationError):
        return [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    # docstrings count as code lines here; reviewers see the same rule in the rubric
    return sorted(lines)


def _python_lexical(src: str) -> dict:
    code_lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    m = re.search(r"def\s+\w+\s*\((.*?)\)\s*(->[^:]*)?:", src, re.S)
    params = _split_top_level(m.group(1)) if m else []
    params = [p for p in params if p not in {"*", "/"}]
    if params and params[0].split(":")[0].strip() in {"self", "cls"}:
        params = params[1:]
    depth, stack = 0, []
    for l in code_lines[1:]:
        ind = len(l) - len(l.lstrip())
        while stack and ind <= stack[-1]:
            stack.pop()
        if re.match(r"\s*(if|elif|else|for|while|try|except|finally|with|match|async\s+for|async\s+with)\b", l):
            stack.append(ind)
            depth = max(depth, len(stack))
    return {"metrics_version": METRICS_VERSION, "effective_loc": len(code_lines),
            "parameter_count": len(params), "max_nesting": depth, "parser": "lexical"}
