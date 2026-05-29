"""Weight: structural cosine similarity of symbol vectors.

For each file, build a feature vector from SSAT:
  - Types it DEFINES (class names, global_code names)
  - Types it REFERENCES (parameter types)
  - Function names it defines

All symbols share the same key space (no def:/ref: prefix split),
so A defining 'Node' and B referencing 'Node' as a param type
properly contribute to the dot product.

Cosine similarity ∈ [0,1], scaled to [0,10] without global-max
normalization so weights are comparable across projects.
"""
import math
from collections import Counter
from .common import extract_files, extract_edges

_BUILTIN_TYPES = {"str", "int", "float", "bool", "bytes", "list", "dict", "set",
                  "tuple", "None", "Any", "Optional", "object", "type"}
_DUNDER = {"__init__", "__repr__", "__str__", "__eq__", "__lt__", "__le__",
           "__gt__", "__ge__", "__hash__", "__len__", "__bool__"}


def _build_symbol_vector(file_item: dict) -> Counter:
    vec = Counter()
    # Defined symbols: classes, top-level functions, global_code (higher weight)
    for cls in file_item.get("classes", []):
        name = cls.get("name", "")
        if name:
            vec[name] += 2
            for fn in cls.get("functions", []):
                fn_name = fn.get("name", "")
                if fn_name and fn_name not in _DUNDER:
                    vec[fn_name] += 1
    for fn in file_item.get("functions", []):
        if fn.get("name", ""):
            vec[fn["name"]] += 2
    for gc in file_item.get("global_code", []):
        if gc.get("name", ""):
            vec[gc["name"]] += 2
    # Referenced types in parameter annotations — same key space as definitions
    # so A defining 'Node' (key='Node', w=2) and B referencing 'Node' (key='Node', w=1)
    # properly contribute to the dot product.
    for cls in file_item.get("classes", []):
        for fn in cls.get("functions", []):
            for p in fn.get("parameters", []):
                ptype = p.get("type", "")
                if ptype and ptype not in _BUILTIN_TYPES:
                    vec[ptype] += 1
    for fn in file_item.get("functions", []):
        for p in fn.get("parameters", []):
            ptype = p.get("type", "")
            if ptype and ptype not in _BUILTIN_TYPES:
                vec[ptype] += 1
    return vec


def _cosine_sim(v1: Counter, v2: Counter) -> float:
    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def weights_ssat_cosine(ssat: list[dict], default_weight: float = 0.1) -> dict[tuple, float]:
    files = extract_files(ssat)
    edges = extract_edges(ssat)
    vectors = {p: _build_symbol_vector(item) for p, item in files.items()
               if not p.endswith("__init__.py")}
    weights = {}
    for source, target in edges:
        sim = _cosine_sim(vectors.get(source, Counter()), vectors.get(target, Counter()))
        # Scale [0,1] → [0,10] using absolute cosine (no max-normalization)
        # so weights are comparable across projects.
        weights[(source, target)] = sim * 10 if sim > 0 else default_weight
    return weights
