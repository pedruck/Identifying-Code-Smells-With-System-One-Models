"""Method/function extraction from source files (frozen extraction policy, ``extract/1``).

Python uses ``ast`` spans.  Brace languages use a comment/string-aware scan for
``name(params) ... {`` headers whose name is not a control keyword, then take
the balanced body.  Extracted snippets are the whole function including its
signature; no enclosing context is added (plan section 5: identical policy
across splits).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .static_metrics import CONTROL_KEYWORDS, strip_comments_and_strings

EXTRACTION_VERSION = "extract/1"
EXTENSIONS = {".py": "python", ".java": "java", ".js": "javascript", ".ts": "typescript", ".cpp": "cpp",
              ".cc": "cpp", ".cxx": "cpp", ".cs": "csharp", ".kt": "kotlin", ".go": "go"}
_NOT_FUNCTIONS = CONTROL_KEYWORDS | {"return", "new", "catch", "sizeof", "throw", "function", "class"}


@dataclass
class Function:
    name: str
    start_line: int     # 1-based, inclusive
    end_line: int
    code: str


def functions(source: str, language: str) -> list[Function]:
    return _python(source) if language == "python" else _brace(source, language)


def _python(source: str) -> list[Function]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            out.append(Function(n.name, start, n.end_lineno, "\n".join(lines[start - 1:n.end_lineno])))
    return sorted(out, key=lambda f: f.start_line)


_HEADER = re.compile(r"([A-Za-z_$~][A-Za-z0-9_$]*(?:::[A-Za-z_~][A-Za-z0-9_]*)?)\s*\(")


def _brace(source: str, language: str) -> list[Function]:
    s = strip_comments_and_strings(source, language)
    # the stripper preserves newlines, so line numbers in ``s`` match the original source
    src_lines = source.split("\n")
    out, i = [], 0
    while True:
        m = _HEADER.search(s, i)
        if not m:
            break
        name = m.group(1).split("::")[-1]
        j, depth = m.end() - 1, 0
        while j < len(s):                        # match the parameter list
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        k = j + 1
        head = re.match(r"[^;{}()=]*?(=>\s*)?\{", s[k:])  # throws/const/-> T/: T before the body
        before = s[max(0, m.start() - 256):m.start()].rstrip()
        is_call = bool(before) and before[-1] in "=.(,!&|?+-*/"   # a call expression, not a declaration
        if head and name not in _NOT_FUNCTIONS and not is_call:
            b = k + head.end() - 1
            depth, e = 0, b
            while e < len(s):
                if s[e] == "{":
                    depth += 1
                elif s[e] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            start_line = s.count("\n", 0, s.rfind("\n", 0, m.start()) + 1) + 1
            end_line = s.count("\n", 0, e) + 1
            out.append(Function(name, start_line, end_line, "\n".join(src_lines[start_line - 1:end_line])))
            i = e + 1                            # nested functions are part of their parent
        else:
            i = m.end()
    return out
