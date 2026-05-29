from __future__ import annotations

import ast


def extract_api(source: str, file_path: str = "<string>") -> str:
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return ""

    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sig = get_signature(node)
            ret = get_return_type(node)
            out.append(f"{prefix} {sig}:\n    \"\"\"returns: {ret}\"\"\"\n")
        elif isinstance(node, ast.ClassDef):
            _extract_class(node, indent=0, out=out)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value_repr = _safe_unparse(node.value)
                    out.append(f"{target.id} = {value_repr}\n")
        elif isinstance(node, ast.AnnAssign) and node.target and isinstance(node.target, ast.Name):
            ann = ast.unparse(node.annotation)
            if node.value:
                value_repr = _safe_unparse(node.value)
                out.append(f"{node.target.id}: {ann} = {value_repr}\n")
            else:
                out.append(f"{node.target.id}: {ann}\n")
    return "\n".join(out)


def _extract_class(node: ast.ClassDef, indent: int, out: list[str]) -> None:
    pad = "    " * indent
    inner = "    " * (indent + 1)
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    class_header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    out.append(f"{pad}{class_header}\n")
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dec = _decorator_prefix(sub)
            prefix = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
            sig = get_signature(sub)
            ret = get_return_type(sub)
            dec_line = f"{dec}\n{inner}" if dec else ""
            out.append(f"{inner}{dec_line}{prefix} {sig}:\n{inner}    \"\"\"returns: {ret}\"\"\"\n")
        elif isinstance(sub, ast.ClassDef):
            _extract_class(sub, indent + 1, out)


def _decorator_prefix(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name) and dec.id in ("property", "staticmethod", "classmethod"):
            return f"@{dec.id}"
        if isinstance(dec, ast.Attribute) and dec.attr == "setter":
            if isinstance(dec.value, ast.Name):
                return f"@{dec.value.id}.setter"
    return ""


def _safe_unparse(node: ast.expr) -> str:
    """Unparse a value node, truncating long literals."""
    try:
        text = ast.unparse(node)
        if len(text) > 80:
            return text[:77] + "..."
        return text
    except Exception:
        return "..."


def get_signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: list[str] = []
    args = func.args

    # positional-only args
    num_posonlyargs = len(args.posonlyargs)
    # defaults align to the right of (posonlyargs + args)
    all_positional = args.posonlyargs + args.args
    num_positional = len(all_positional)
    num_defaults = len(args.defaults)
    default_offset = num_positional - num_defaults

    for i, arg in enumerate(all_positional):
        part = _format_arg(arg)
        if i >= default_offset:
            default = args.defaults[i - default_offset]
            part += f"={_safe_unparse(default)}"
        parts.append(part)
        if i == num_posonlyargs - 1 and num_posonlyargs > 0:
            parts.append("/")

    # *args or bare *
    if args.vararg:
        parts.append(f"*{_format_arg(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    # keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        part = _format_arg(arg)
        kw_def = args.kw_defaults[i] if i < len(args.kw_defaults) else None
        if kw_def is not None:
            part += f"={_safe_unparse(kw_def)}"
        parts.append(part)

    # **kwargs
    if args.kwarg:
        parts.append(f"**{_format_arg(args.kwarg)}")

    return f"{func.name}({', '.join(parts)})"


def _format_arg(arg: ast.arg) -> str:
    if arg.annotation:
        return f"{arg.arg}: {ast.unparse(arg.annotation)}"
    return arg.arg


def get_return_type(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if func.returns:
        return ast.unparse(func.returns)
    return infer_return_type(func)


def infer_return_type(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.Constant):
                return type(node.value.value).__name__
            if isinstance(node.value, ast.Name):
                return "UnknownVariable"
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                return f"{node.value.func.id}()"
    return "None"
