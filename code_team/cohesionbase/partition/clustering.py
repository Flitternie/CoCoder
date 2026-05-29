"""Shared clustering algorithms — Louvain, directed Louvain, InfoMap, Leiden.

All functions take (weights, nodes, **kwargs) → partition dict.
"""
from .common import louvain_partition  # undirected Louvain (already in common.py)


def directed_louvain(weights: dict[tuple, float], nodes: set[str], resolution: float = 1.0) -> dict[str, int]:
    """Louvain on directed graph — preserves import direction."""
    import networkx as nx
    G = nx.DiGraph()
    for f in nodes: G.add_node(f)
    for (s, t), w in weights.items():
        if w > 0 and s in nodes and t in nodes:
            G.add_edge(s, t, weight=w)
    if G.number_of_edges() == 0:
        return {f: i for i, f in enumerate(sorted(nodes))}
    communities = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=42)
    partition = {}
    for i, comm in enumerate(communities):
        for f in comm: partition[f] = i
    return partition


def infomap_partition(weights: dict[tuple, float], nodes: set[str]) -> dict[str, int]:
    """InfoMap — random walk based, natively directed. (Rosvall & Bergstrom 2008)"""
    from infomap import Infomap
    node_list = sorted(nodes)
    if not node_list:
        return {}
    edge_list = [
        (s, t, w) for (s, t), w in weights.items()
        if w > 0 and s in nodes and t in nodes
    ]
    if not edge_list:
        return {f: i for i, f in enumerate(node_list)}
    node_to_id = {f: i for i, f in enumerate(node_list)}
    im = Infomap(silent=True, directed=True, seed=42)
    for f in node_list: im.add_node(node_to_id[f], name=f)
    for s, t, w in edge_list:
        im.add_link(node_to_id[s], node_to_id[t], w)
    im.run()
    partition = {}
    for node_id in im.tree:
        if node_id.is_leaf:
            partition[node_list[node_id.node_id]] = node_id.module_id
    return partition


def leiden_partition(weights: dict[tuple, float], nodes: set[str], resolution: float = 1.0) -> dict[str, int]:
    """Leiden — improved Louvain with guaranteed well-connected communities. (Traag et al. 2019)"""
    import igraph as ig
    import leidenalg
    node_list = sorted(nodes)
    node_to_id = {f: i for i, f in enumerate(node_list)}
    G = ig.Graph(directed=True)
    G.add_vertices(len(node_list))
    G.vs["name"] = node_list
    edge_list, weight_list = [], []
    for (s, t), w in weights.items():
        if w > 0 and s in nodes and t in nodes:
            edge_list.append((node_to_id[s], node_to_id[t]))
            weight_list.append(w)
    if not edge_list:
        return {f: i for i, f in enumerate(node_list)}
    G.add_edges(edge_list)
    G.es["weight"] = weight_list
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition,
                                    weights="weight", resolution_parameter=resolution, seed=42)
    partition = {}
    for i, comm in enumerate(part):
        for nid in comm: partition[node_list[nid]] = i
    return partition
