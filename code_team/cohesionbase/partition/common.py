"""Common utilities: data loading, role detection, evaluation, Louvain."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


# --- Data Loading ---

def load_rib(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

def extract_files(rib: list[dict]) -> dict[str, dict]:
    files = {}
    for m in rib:
        for f in m.get("files", []):
            p = f.get("path", "")
            if p and p.endswith(".py"): files[p] = f
    return files

def extract_edges(rib: list[dict]) -> list[tuple[str, str]]:
    files = extract_files(rib)
    return [(d, f.get("path", "")) for m in rib for f in m.get("files", [])
            for d in f.get("dependencies", []) if d in files and d != f.get("path", "")]

def load_project(project: str) -> tuple[list[dict], set[str], list[tuple]]:
    """Load RIB, return (rib, non_init_files, non_init_edges)."""
    rib = load_rib(DATA_DIR / project / "rib.json")
    files = extract_files(rib)
    non_init = {f for f in files if not f.endswith("__init__.py")}
    edges = extract_edges(rib)
    ni_edges = [(s, t) for s, t in edges if s in non_init and t in non_init]
    return rib, non_init, ni_edges


# --- Role Detection ---

def detect_roles(rib: list[dict], threshold: float = 0.5) -> dict[str, str]:
    """in_hub (high in-degree), out_hub (high out-degree), core (rest)."""
    files = extract_files(rib)
    edges = extract_edges(rib)
    non_init = {f for f in files if not f.endswith("__init__.py")}
    n = len(non_init)
    if n <= 1: return {f: "core" for f in non_init}

    deps_of = defaultdict(set)
    depended_by = defaultdict(set)
    for s, t in edges:
        if s in non_init and t in non_init:
            deps_of[t].add(s); depended_by[s].add(t)

    roles = {}
    for f in non_init:
        fi, fo = len(deps_of[f]) / (n - 1), len(depended_by[f]) / (n - 1)
        if fi > threshold: roles[f] = "out_hub"
        elif fo > threshold: roles[f] = "in_hub"
        else: roles[f] = "core"
    return roles


# --- Louvain ---

def louvain_partition(weights: dict[tuple, float], all_files: set[str], resolution: float = 1.0) -> dict[str, int]:
    import community as cl; import networkx as nx
    G = nx.Graph()
    for f in all_files: G.add_node(f)
    for (s, t), w in weights.items():
        if w > 0:
            ex = G.get_edge_data(s, t)
            G.add_edge(s, t, weight=max(w, ex["weight"] if ex else 0))
    if G.number_of_edges() == 0:
        return {f: i for i, f in enumerate(sorted(all_files))}
    return cl.best_partition(G, weight="weight", resolution=resolution, random_state=42)


# --- Role + Louvain Grouping ---

def grouping_role_louvain(rib: list[dict], weight_dict: dict[tuple, float], resolution: float = 1.0) -> dict[str, int]:
    files = extract_files(rib)
    roles = detect_roles(rib)
    in_hubs = sorted(f for f, r in roles.items() if r == "in_hub")
    out_in_hubs = sorted(f for f, r in roles.items() if r == "out_hub")
    cores = sorted(f for f, r in roles.items() if r == "core")
    core_set = set(cores)
    core_w = {k: v for k, v in weight_dict.items() if k[0] in core_set and k[1] in core_set}
    core_part = louvain_partition(core_w, core_set, resolution=resolution)

    partition = {}; gid = 0
    for f in in_hubs: partition[f] = gid; gid += 1
    cg = defaultdict(list)
    for f, cid in core_part.items(): cg[cid].append(f)
    for _, gf in sorted(cg.items()):
        for f in gf: partition[f] = gid
        gid += 1
    init_files = [f for f in files if f.endswith("__init__.py")]
    for f in out_hubs: partition[f] = gid
    for f in init_files: partition[f] = gid
    if out_hubs or init_files: gid += 1
    return partition


# --- Evaluation ---

def partition_to_groups(p: dict[str, int]) -> list[list[str]]:
    g = defaultdict(list)
    for f, gid in p.items(): g[gid].append(f)
    return [sorted(v) for v in sorted(g.values(), key=lambda x: x[0])]

def cross_group_edges(p: dict[str, int], edges: list[tuple]) -> int:
    return sum(1 for s, t in edges if p.get(s) != p.get(t))

def rand_index(pred: dict[str, int], truth: dict[str, int]) -> float:
    c = sorted(set(pred) & set(truth))
    if len(c) < 2: return 0.0
    a = t = 0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if (pred[c[i]] == pred[c[j]]) == (truth[c[i]] == truth[c[j]]): a += 1
            t += 1
    return a / t if t else 0.0

GROUND_TRUTH = {
    "bplustree": [
        ["bplustree/const.py"],
        ["bplustree/entry.py", "bplustree/node.py", "bplustree/memory.py"],
        ["bplustree/serializer.py", "bplustree/utils.py"],
        ["bplustree/tree.py", "bplustree/__init__.py"],
    ],
    "cookiecutter": [
        ["cookiecutter/exceptions.py"],
        ["cookiecutter/extensions.py"],
        ["cookiecutter/environment.py", "cookiecutter/utils.py"],
        ["cookiecutter/config.py"],
        ["cookiecutter/find.py"],
        ["cookiecutter/__init__.py"],
        ["cookiecutter/log.py"],
        ["cookiecutter/prompt.py"],
        ["cookiecutter/hooks.py"],
        ["cookiecutter/replay.py"],
        ["cookiecutter/generate.py"],
        ["cookiecutter/vcs.py", "cookiecutter/zipfile.py", "cookiecutter/repository.py"],
        ["cookiecutter/main.py", "cookiecutter/cli.py", "cookiecutter/__main__.py"],
    ],
    "imapclient": [
        ["imapclient/exceptions.py"],
        ["imapclient/typing_imapclient.py"],
        ["imapclient/version.py"],
        ["imapclient/imap4.py"],
        ["imapclient/imap_utf7.py"],
        ["imapclient/tls.py"],
        ["imapclient/fixed_offset.py", "imapclient/datetime_util.py"],
        ["imapclient/util.py"],
        ["imapclient/response_lexer.py", "imapclient/response_types.py"],
        ["imapclient/response_parser.py", "imapclient/imapclient.py"],
        ["imapclient/interact.py", "imapclient/testable_imapclient.py", "imapclient/config.py", "imapclient/__init__.py"],
    ],
    "simpy": [
        ["simpy/exceptions.py", "simpy/events.py", "simpy/core.py"],
        ["simpy/resources/__init__.py", "simpy/resources/base.py"],
        ["simpy/rt.py"],
        ["simpy/util.py"],
        ["simpy/resources/container.py"],
        ["simpy/resources/resource.py"],
        ["simpy/resources/store.py"],
        ["simpy/__init__.py"],
    ],
}

def truth_partition(proj: str) -> dict[str, int]:
    p = {}
    for i, g in enumerate(GROUND_TRUTH.get(proj, [])):
        for f in g: p[f] = i
    return p

def evaluate(name: str, partition: dict[str, int], ni_edges: list[tuple], truth: dict[str, int], verbose: bool = True) -> dict:
    groups = partition_to_groups(partition)
    ri = rand_index(partition, truth)
    xe = cross_group_edges(partition, ni_edges)
    if verbose:
        print(f"  {name:<40s} Rand={ri:.3f}  Groups={len(groups):2d}  XEdge={xe:2d}")
        for g in groups: print(f"    {[f.split('/')[-1] for f in g]}")
    return {"rand": ri, "groups": len(groups), "xe": xe}


# --- LLM Client ---

def llm_client():
    from openai import OpenAI
    ak = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    bu = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    pv = os.environ.get("PORTKEY_PROVIDER", "")
    return OpenAI(api_key=ak, base_url=bu, default_headers={"x-portkey-provider": pv} if pv else {})

def llm_model():
    m = os.environ.get("LLM_MODEL", "gpt-5-mini")
    return m.split("/", 1)[-1] if "/" in m else m
