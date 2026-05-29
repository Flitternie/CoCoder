"""Shared post-processing: Role Detection grouping + sibling split + merge.

role_grouping: remove hub/integration → run algorithm on core → reassemble
lift_independent: split groups where files share parent but don't depend on each other
merge_small_groups: merge tiny adjacent groups only when parallel makespan is unchanged
attach_init_files: attach __init__.py files after partitioning without affecting it
"""
from collections import defaultdict
from pathlib import Path
from .common import extract_files, extract_edges, detect_roles, louvain_partition


def role_grouping(rib, weights, algo_fn, threshold=0.4, **algo_kwargs):
    """Role Detection + algorithm on core files + reassemble non-init files."""
    roles = detect_roles(rib, threshold=threshold)
    in_hubs = sorted(f for f, r in roles.items() if r == "in_hub")
    out_in_hubs = sorted(f for f, r in roles.items() if r == "out_hub")
    cores = sorted(f for f, r in roles.items() if r == "core")
    core_set = set(cores)
    core_w = {k: v for k, v in weights.items() if k[0] in core_set and k[1] in core_set}

    core_part = algo_fn(core_w, core_set, **algo_kwargs)

    partition = {}; gid = 0
    for f in in_hubs: partition[f] = gid; gid += 1
    cg = defaultdict(list)
    for f, cid in core_part.items(): cg[cid].append(f)
    for _, gf in sorted(cg.items()):
        for f in gf: partition[f] = gid
        gid += 1
    for f in out_hubs: partition[f] = gid
    if out_hubs: gid += 1
    return partition


def lift_independent(partition: dict[str, int], rib: list[dict]) -> dict[str, int]:
    """Split groups where sibling files share a parent but don't depend on each other."""
    edges = extract_edges(rib)
    deps_of = defaultdict(set)
    for s, t in edges: deps_of[t].add(s)

    groups = defaultdict(list)
    for f, gid in partition.items(): groups[gid].append(f)

    new_partition = {}; next_gid = 0

    for gid, files in sorted(groups.items()):
        if len(files) <= 2:
            for f in files: new_partition[f] = next_gid
            next_gid += 1
            continue

        file_set = set(files)
        internal_deps = defaultdict(set)
        internal_depended_by = defaultdict(set)
        for f in files:
            for dep in deps_of.get(f, set()):
                if dep in file_set:
                    internal_deps[f].add(dep)
                    internal_depended_by[dep].add(f)

        internal_hubs = {f for f in files if len(internal_depended_by.get(f, set())) >= 2}

        if not internal_hubs:
            # No hub: split by connected components
            adj = defaultdict(set)
            for f in files:
                for dep in internal_deps.get(f, set()):
                    adj[f].add(dep); adj[dep].add(f)
            visited = set()
            for f in sorted(files):
                if f in visited: continue
                comp = []
                queue = [f]
                while queue:
                    node = queue.pop(0)
                    if node in visited: continue
                    visited.add(node); comp.append(node)
                    for nb in adj.get(node, set()):
                        if nb not in visited: queue.append(nb)
                for cf in comp: new_partition[cf] = next_gid
                next_gid += 1
        else:
            # Has hub: split siblings (files only depending on hub, no dependents)
            siblings, chain_files = [], set()
            for f in files:
                if f in internal_hubs:
                    chain_files.add(f); continue
                f_deps = internal_deps.get(f, set())
                f_dependents = internal_depended_by.get(f, set())
                if f_deps and f_deps.issubset(internal_hubs) and not f_dependents:
                    siblings.append(f)
                else:
                    chain_files.add(f)

            if chain_files:
                for f in sorted(chain_files): new_partition[f] = next_gid
                next_gid += 1
            for f in sorted(siblings):
                new_partition[f] = next_gid; next_gid += 1
            for f in files:
                if f not in new_partition:
                    new_partition[f] = next_gid; next_gid += 1

    return new_partition


def _compute_file_waves(
    edges: list[tuple[str, str]], file_set: set[str],
) -> dict[str, int]:
    """Compute topological wave layer for each file. wave(f) = max(wave(deps)) + 1."""
    deps_of: dict[str, set[str]] = defaultdict(set)
    for dep, f in edges:
        if dep in file_set and f in file_set:
            deps_of[f].add(dep)
    layer: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for f in file_set:
            if f in layer:
                continue
            deps = deps_of.get(f, set())
            if all(d in layer for d in deps):
                layer[f] = max([0] + [layer[d] + 1 for d in deps])
                changed = True
    for f in file_set:
        layer.setdefault(f, 0)
    return layer


def _estimate_file_work(rib: list[dict]) -> dict[str, float]:
    """Estimate relative implementation work from RIB symbol counts."""
    work = {}
    for path, item in extract_files(rib).items():
        count = 0
        for cls in item.get("classes", []):
            count += 2
            count += len(cls.get("functions", []))
        count += len(item.get("functions", []))
        count += len(item.get("global_code", []))
        work[path] = max(float(count), 1.0)
    return work


def _simulate_zero_comm_makespan(
    file_set: set[str],
    edges: list[tuple[str, str]],
    work: dict[str, float],
    partition: dict[str, int],
) -> float:
    """Simulate parallel execution with group serialization and no comm penalty."""
    successors: dict[str, list[str]] = defaultdict(list)
    predecessors: dict[str, list[str]] = defaultdict(list)
    in_degree = {f: 0 for f in file_set}
    for dep, f in edges:
        if dep in file_set and f in file_set:
            successors[dep].append(f)
            predecessors[f].append(dep)
            in_degree[f] += 1

    topo = []
    queue = sorted(f for f in file_set if in_degree[f] == 0)
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for succ in sorted(successors.get(node, [])):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                queue.sort()

    # Some analyzed projects contain import cycles. Keep the simulation
    # deterministic by appending cyclic leftovers instead of failing closed.
    topo.extend(sorted(file_set - set(topo)))

    finish: dict[str, float] = {}
    group_free: dict[int, float] = defaultdict(float)
    for f in topo:
        gid = partition[f]
        start = group_free[gid]
        for pred in predecessors.get(f, []):
            start = max(start, finish.get(pred, 0.0))
        finish[f] = start + work.get(f, 1.0)
        group_free[gid] = finish[f]

    return max(finish.values()) if finish else 0.0


def _build_reachability(
    edges: list[tuple[str, str]], file_set: set[str],
) -> dict[str, set[str]]:
    successors: dict[str, list[str]] = defaultdict(list)
    for dep, f in edges:
        if dep in file_set and f in file_set:
            successors[dep].append(f)

    reach = {}
    for f in file_set:
        seen = set()
        stack = list(successors.get(f, []))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(successors.get(node, []))
        reach[f] = seen
    return reach


def attach_init_files(
    partition: dict[str, int],
    rib: list[dict],
    roles: dict[str, str] | None = None,
) -> tuple[dict[str, int], list[dict]]:
    """Attach __init__.py files after partitioning visible files.

    The returned partition includes init files for ownership/output purposes,
    but the assignment is metadata: init files do not affect role grouping,
    sibling splitting, or merge decisions.
    """
    files = extract_files(rib)
    result = dict(partition)
    attachments = []
    roles = roles or detect_roles(rib)

    def _group_name(gid: int) -> str:
        return f"group_{gid}"

    def _choose_group(candidates: list[str]) -> tuple[int | None, list[str]]:
        by_group: dict[int, list[str]] = defaultdict(list)
        for f in candidates:
            if f in result:
                by_group[result[f]].append(f)
        if not by_group:
            return None, []
        gid, via = sorted(
            by_group.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[0]
        return gid, sorted(via)

    out_hub_files = [
        f for f, role in roles.items()
        if role == "out_hub" and f in result
    ]
    out_hub_gid, out_hub_via = _choose_group(out_hub_files)
    next_gid = (max(result.values()) + 1) if result else 0

    for init_path in sorted(f for f in files if f.endswith("__init__.py")):
        item = files[init_path]
        deps = [
            d for d in item.get("dependencies", [])
            if d in result and not d.endswith("__init__.py")
        ]
        gid, via = _choose_group(deps)
        reason = "dependency" if gid is not None else ""

        parts = Path(init_path).parts
        is_root_init = len(parts) == 2
        if gid is None and is_root_init and out_hub_gid is not None:
            gid = out_hub_gid
            via = out_hub_via
            reason = "root_out_hub"

        if gid is None:
            parent = str(Path(init_path).parent)
            same_dir = [
                f for f in result
                if not f.endswith("__init__.py")
                and str(Path(f).parent) == parent
            ]
            gid, via = _choose_group(same_dir)
            if gid is not None:
                reason = "same_dir"

        if gid is None:
            gid = next_gid
            next_gid += 1
            via = []
            reason = "hidden_init"

        result[init_path] = gid
        attachments.append({
            "file": init_path,
            "attached_to_group": _group_name(gid),
            "reason": reason,
            "via": via,
        })

    return result, attachments


def merge_small_groups(
    partition: dict[str, int],
    rib: list[dict],
    weights: dict[tuple[str, str], float] | None = None,
    max_group_size: int = 8,
    max_wave_gap: int | None = None,
) -> dict[str, int]:
    """Bottom-up dependency merge that preserves zero-communication makespan.

    __init__.py files are excluded when counting group size.
    A group may only merge into a group it depends on, following edges in the
    reverse direction: dep -> file means group(file) may merge into group(dep).
    A merge is accepted only if the merged partition has the same pure parallel
    makespan as the current partition when communication cost is zero.
    """
    edges = extract_edges(rib)
    file_set = set(partition)
    file_waves = _compute_file_waves(edges, file_set)
    work = _estimate_file_work(rib)
    reachability = _build_reachability(edges, file_set)

    def _real_size(files):
        return sum(1 for f in files if not f.endswith("__init__.py"))

    def _wave_gap(files_a, files_b):
        wa = [file_waves[f] for f in files_a if not f.endswith("__init__.py")]
        wb = [file_waves[f] for f in files_b if not f.endswith("__init__.py")]
        if not wa or not wb:
            return 0
        return min(abs(a - b) for a in wa for b in wb)

    def _merged_partition(source_gid, target_gid):
        return {f: (target_gid if g == source_gid else g) for f, g in partition.items()}

    def _removed_coupling(source_gid, target_gid):
        total = 0.0
        for dep, f in edges:
            if dep in partition and f in partition:
                gd, gf = partition[dep], partition[f]
                if {gd, gf} == {source_gid, target_gid}:
                    total += (weights or {}).get((dep, f), 1.0)
        return total

    def _chain_compatible(files_a, files_b):
        real_a = [f for f in files_a if not f.endswith("__init__.py")]
        real_b = [f for f in files_b if not f.endswith("__init__.py")]
        for a in real_a:
            for b in real_b:
                if a == b:
                    continue
                if b not in reachability[a] and a not in reachability[b]:
                    return False
        return True

    while True:
        groups: dict[int, list[str]] = defaultdict(list)
        for f, gid in partition.items():
            groups[gid].append(f)

        eligible_sources = {
            gid for gid, files in groups.items()
            if _real_size(files) < max_group_size
        }
        if not eligible_sources:
            break

        # Only consider bottom-up merges: dependent group -> dependency group.
        pair_edges: dict[tuple[int, int], int] = defaultdict(int)
        for dep, f in edges:
            if dep in partition and f in partition:
                dep_gid, file_gid = partition[dep], partition[f]
                if dep_gid != file_gid and file_gid in eligible_sources:
                    pair_edges[(file_gid, dep_gid)] += 1

        current_makespan = _simulate_zero_comm_makespan(file_set, edges, work, partition)

        # Find the deepest safe merge, so chains can collapse bottom-up.
        best, best_key = None, None
        for (S, N), ec in pair_edges.items():
            if _real_size(groups[S]) + _real_size(groups[N]) > max_group_size:
                continue
            if not _chain_compatible(groups[S], groups[N]):
                continue
            gap = _wave_gap(groups[S], groups[N])
            if max_wave_gap is not None and gap > max_wave_gap:
                continue
            candidate = _merged_partition(S, N)
            candidate_makespan = _simulate_zero_comm_makespan(
                file_set, edges, work, candidate
            )
            if candidate_makespan > current_makespan + 1e-9:
                continue
            removed = _removed_coupling(S, N)
            merged_size = _real_size(groups[S]) + _real_size(groups[N])
            source_wave = max(file_waves.get(f, 0) for f in groups[S])
            key = (-source_wave, candidate_makespan, -removed, -ec, merged_size, gap)
            if best_key is None or key < best_key:
                best, best_key = (S, N), key

        if best is None:
            break

        S, N = best
        partition = _merged_partition(S, N)

    return partition
