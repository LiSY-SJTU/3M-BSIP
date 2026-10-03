from collections import defaultdict
from dataclasses import dataclass
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


DNAAtom = Tuple[str, int, str, str, np.ndarray]

DNA_BACKBONE_ATOMS = frozenset({
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
    "C2'", "C1'",
})
DNA_SELF_CLASH_THRESHOLD = 1.5


def _normalize_atom_name(name: str) -> str:
    normalized = name.strip().upper().replace("*", "'")
    return {"O1P": "OP1", "O2P": "OP2"}.get(normalized, normalized)


@dataclass
class Placement:
    protein_chain_id: str
    protein_resseq: int
    aa_type: str
    template_id: str
    rmsd_to_target: float
    is_specific: bool
    dna_atoms: List[DNAAtom]

    def chain_resseq_order(self) -> Dict[str, List[int]]:
        order: Dict[str, List[int]] = {}
        chains = sorted({c for (c, _, _, _, _) in self.dna_atoms})
        for ch in chains:
            order[ch] = sorted({
                resseq for (c, resseq, _, _, _) in self.dna_atoms if c == ch
            })
        return order

    def has_complete_dna_fragment(self, chains: List[str]) -> bool:
        """Return whether this placement can be used by the 3-mer assembler."""
        order = self.chain_resseq_order()
        return all(len(order.get(ch, [])) == 3 for ch in chains)


@dataclass(frozen=True)
class CompatibilityEdge:
    source: int
    target: int
    offset: int
    backbone_rmsd: float
    common_atoms: int
    chain_common_atoms: Tuple[int, int]


class OverlapGraph(list):
    """Positive-extension adjacency plus all signed compatibility metadata.

    It remains a list subclass so existing callers that inspect or compare the
    old adjacency-list return value continue to work.  ``edges[(u, v)]`` also
    records zero-offset consensus and the inverse negative-offset relations.
    """

    def __init__(
        self,
        adjacency: Sequence[Sequence[int]],
        edges: Mapping[Tuple[int, int], CompatibilityEdge],
        valid_nodes: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__([list(neighbors) for neighbors in adjacency])
        self.edges = dict(edges)
        self.valid_nodes = frozenset(
            range(len(adjacency)) if valid_nodes is None else valid_nodes
        )

    def edge(self, source: int, target: int) -> Optional[CompatibilityEdge]:
        return self.edges.get((source, target))


class AssemblyPath(list):
    """Selected placements with an integer 3-mer start for every list item."""

    def __init__(
        self, nodes: Iterable[int] = (), coordinates: Iterable[int] = (), score: float = 0.0,
    ) -> None:
        super().__init__(nodes)
        self.coordinates = tuple(coordinates)
        self.score = float(score)


def _residue_atom_maps(
    placement: Placement, chain: str, backbone_only: bool = True,
) -> List[Dict[str, np.ndarray]]:
    residue_order = placement.chain_resseq_order().get(chain, [])
    result: List[Dict[str, np.ndarray]] = []
    for resseq in residue_order:
        atom_map: Dict[str, np.ndarray] = {}
        for c, r, _, atom_name, xyz in placement.dna_atoms:
            if c != chain or r != resseq:
                continue
            normalized = _normalize_atom_name(atom_name)
            if backbone_only and normalized not in DNA_BACKBONE_ATOMS:
                continue
            atom_map.setdefault(normalized, xyz)
        result.append(atom_map)
    return result


def dna_atoms_internal_clash_count(
    dna_atoms: Sequence[DNAAtom],
    chains: List[str],
    threshold: float = DNA_SELF_CLASH_THRESHOLD,
) -> int:
    """Count impossible nonlocal heavy-atom contacts inside a DNA fragment.

    Contacts within one residue and between adjacent residues on one strand
    are covalent/local and are excluded.  Every cross-strand pair is tested.
    Residue/base identities do not enter this geometric validity check.
    """
    chain_order = {
        chain: sorted({
            resseq for atom_chain, resseq, _, _, _ in dna_atoms
            if atom_chain == chain
        })
        for chain in chains
    }
    if not all(len(chain_order[chain]) == 3 for chain in chains):
        return 0
    residue_slots = {
        (chain, resseq): slot
        for chain in chains
        for slot, resseq in enumerate(chain_order[chain])
    }
    residue_coordinates: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for chain, resseq, _, atom_name, xyz in dna_atoms:
        if chain not in chains or (chain, resseq) not in residue_slots:
            continue
        normalized = _normalize_atom_name(atom_name).lstrip("0123456789")
        if not normalized or normalized.startswith(("H", "D")):
            continue
        residue_coordinates[(chain, residue_slots[(chain, resseq)])].append(xyz)

    residue_keys = sorted(residue_coordinates)
    clashes = 0
    for left_index, (left_chain, left_slot) in enumerate(residue_keys):
        left_coordinates = np.vstack(residue_coordinates[(left_chain, left_slot)])
        for right_chain, right_slot in residue_keys[left_index + 1:]:
            right_coordinates = np.vstack(
                residue_coordinates[(right_chain, right_slot)]
            )
            if left_chain == right_chain and abs(left_slot - right_slot) <= 1:
                continue
            distances_squared = np.sum(
                (left_coordinates[:, None, :] - right_coordinates[None, :, :]) ** 2,
                axis=2,
            )
            clashes += int(np.count_nonzero(distances_squared < threshold ** 2))
    return clashes


def placement_internal_clash_count(
    placement: Placement,
    chains: List[str],
    threshold: float = DNA_SELF_CLASH_THRESHOLD,
) -> int:
    return dna_atoms_internal_clash_count(placement.dna_atoms, chains, threshold)


def has_valid_internal_dna_geometry(
    placement: Placement,
    chains: List[str],
    threshold: float = DNA_SELF_CLASH_THRESHOLD,
) -> bool:
    return placement_internal_clash_count(placement, chains, threshold) == 0


def _overlap_slot_pairs(offset: int) -> List[Tuple[int, int]]:
    """Return (left, right) local bp slots for a signed start offset."""
    return [
        (right_slot + offset, right_slot)
        for right_slot in range(3)
        if 0 <= right_slot + offset < 3
    ]


def _chain_slot_pairs(
    chain_position: int, offset: int,
) -> List[Tuple[int, int]]:
    pairs = _overlap_slot_pairs(offset)
    if chain_position == 0:
        return pairs
    return [(2 - left, 2 - right) for left, right in pairs]


def _direct_overlap_metrics(
    p: Placement,
    q: Placement,
    chains: List[str],
    offset: int,
    distance_threshold: float,
) -> Optional[Tuple[float, Tuple[int, int]]]:
    counts: List[int] = []
    squared_distances: List[float] = []
    for chain_position, chain in enumerate(chains):
        p_maps = _residue_atom_maps(p, chain)
        q_maps = _residue_atom_maps(q, chain)
        if len(p_maps) != 3 or len(q_maps) != 3:
            return None
        chain_distances: List[float] = []
        close_count = 0
        for p_slot, q_slot in _chain_slot_pairs(chain_position, offset):
            for atom_name in sorted(set(p_maps[p_slot]) & set(q_maps[q_slot])):
                distance = float(np.linalg.norm(
                    p_maps[p_slot][atom_name] - q_maps[q_slot][atom_name]
                ))
                if distance < distance_threshold:
                    close_count += 1
                chain_distances.append(distance)
        minimum = 3 if abs(offset) == 2 else 5
        if close_count < minimum:
            return None
        counts.append(close_count)
        squared_distances.extend(distance * distance for distance in chain_distances)
    overlap_rmsd = math.sqrt(sum(squared_distances) / len(squared_distances))
    rmsd_limit = distance_threshold * (0.75 if abs(offset) == 2 else 1.0)
    if overlap_rmsd > rmsd_limit:
        return None
    return overlap_rmsd, (counts[0], counts[1])


def overlap_pass(
    p: Placement,
    q: Placement,
    chains: List[str],
    distance_threshold: float,
    min_common: int = 5,
) -> bool:
    """Return whether ``q`` is a backbone-compatible +1 extension of ``p``.

    ``min_common`` is retained for source compatibility.  The assembler uses
    a minimum of five close sugar-phosphate atoms independently on each strand.
    """
    if len(chains) != 2:
        raise ValueError("3-mer duplex assembly requires exactly two DNA chains")
    metrics = _direct_overlap_metrics(p, q, chains, 1, distance_threshold)
    if metrics is None:
        return False
    return all(count >= min_common for count in metrics[1])


def build_overlap_graph(
    placements: List[Placement], chains: List[str], threshold: float,
) -> OverlapGraph:
    """Build backbone-only compatibility for offsets 0, +/-1 and +/-2.

    Candidate pairs are generated with spatial joins rather than an O(n^2)
    placement loop.  Positive offsets appear in the list-like adjacency;
    signed and zero-offset relations are available through ``graph.edges``.
    """
    if len(chains) != 2:
        raise ValueError("3-mer duplex assembly requires exactly two DNA chains")
    n = len(placements)
    adjacency: List[List[int]] = [[] for _ in range(n)]
    valid = [
        placement.has_complete_dna_fragment(chains)
        and has_valid_internal_dna_geometry(placement, chains)
        for placement in placements
    ]
    residue_maps = {
        (placement_index, chain): _residue_atom_maps(placement, chain)
        for placement_index, placement in enumerate(placements)
        if valid[placement_index]
        for chain in chains
    }
    edges: Dict[Tuple[int, int], CompatibilityEdge] = {}

    for offset in (0, 1, 2):
        pair_chain_count: Dict[Tuple[int, int, int], int] = defaultdict(int)
        for chain_position, chain in enumerate(chains):
            for p_slot, q_slot in _chain_slot_pairs(chain_position, offset):
                atom_names = sorted(DNA_BACKBONE_ATOMS)
                for atom_name in atom_names:
                    left_atoms = [
                        (index, residue_maps[(index, chain)][p_slot][atom_name])
                        for index in range(n)
                        if valid[index]
                        and atom_name in residue_maps[(index, chain)][p_slot]
                    ]
                    right_atoms = [
                        (index, residue_maps[(index, chain)][q_slot][atom_name])
                        for index in range(n)
                        if valid[index]
                        and atom_name in residue_maps[(index, chain)][q_slot]
                    ]
                    if not left_atoms or not right_atoms:
                        continue
                    left_coords = np.vstack([item[1] for item in left_atoms])
                    right_coords = np.vstack([item[1] for item in right_atoms])
                    neighbors = cKDTree(left_coords).query_ball_tree(
                        cKDTree(right_coords), threshold
                    )
                    for left_position, right_positions in enumerate(neighbors):
                        source = left_atoms[left_position][0]
                        for right_position in right_positions:
                            target = right_atoms[right_position][0]
                            if source == target:
                                continue
                            distance = float(np.linalg.norm(
                                left_coords[left_position] - right_coords[right_position]
                            ))
                            if distance >= threshold:
                                continue
                            pair_chain_count[(source, target, chain_position)] += 1

        minimum = 3 if offset == 2 else 5
        candidate_pairs = sorted({
            (source, target)
            for source, target, _ in pair_chain_count
            if pair_chain_count.get((source, target, 0), 0) >= minimum
            and pair_chain_count.get((source, target, 1), 0) >= minimum
        })
        for source, target in candidate_pairs:
            metrics = _direct_overlap_metrics(
                placements[source], placements[target], chains, offset, threshold
            )
            if metrics is None:
                continue
            backbone_rmsd, chain_counts = metrics
            common_atoms = sum(chain_counts)
            edge = CompatibilityEdge(
                source, target, offset, backbone_rmsd, common_atoms, chain_counts
            )
            old = edges.get((source, target))
            if old is None or (
                edge.backbone_rmsd, abs(edge.offset), edge.offset
            ) < (
                old.backbone_rmsd, abs(old.offset), old.offset
            ):
                edges[(source, target)] = edge
            if offset > 0:
                inverse = CompatibilityEdge(
                    target, source, -offset, backbone_rmsd, common_atoms, chain_counts
                )
                edges[(target, source)] = inverse

    for (source, target), edge in sorted(edges.items()):
        if edge.offset > 0:
            adjacency[source].append(target)
    for neighbors in adjacency:
        neighbors[:] = sorted(set(neighbors))
    return OverlapGraph(
        adjacency, edges,
        valid_nodes=(index for index, is_valid in enumerate(valid) if is_valid),
    )


def _placement_key(placement: Placement, index: int) -> Tuple:
    return (
        float(placement.rmsd_to_target),
        not placement.is_specific,
        placement.protein_chain_id,
        placement.protein_resseq,
        placement.template_id,
        index,
    )


@dataclass(frozen=True)
class _ConsensusGroup:
    members: Tuple[int, ...]
    residue_keys: frozenset
    primary: int
    quality: float


def _consensus_groups(
    graph: OverlapGraph, placements: List[Placement],
) -> Tuple[List[_ConsensusGroup], Dict[int, int]]:
    """Greedily form deterministic groups around direct offset-zero matches."""
    ordered = sorted(
        graph.valid_nodes, key=lambda i: _placement_key(placements[i], i)
    )
    unassigned = set(ordered)
    groups: List[_ConsensusGroup] = []
    node_to_group: Dict[int, int] = {}
    for seed in ordered:
        if seed not in unassigned:
            continue
        direct = [
            node for node in ordered
            if node in unassigned
            and (node == seed or (
                graph.edge(seed, node) is not None
                and graph.edge(seed, node).offset == 0
            ))
        ]
        best_by_residue: Dict[Tuple[str, int], int] = {}
        for node in direct:
            placement = placements[node]
            key = (placement.protein_chain_id, placement.protein_resseq)
            current = best_by_residue.get(key)
            if current is None or _placement_key(placement, node) < _placement_key(
                placements[current], current
            ):
                best_by_residue[key] = node
        members = tuple(sorted(best_by_residue.values(), key=lambda i: _placement_key(
            placements[i], i
        )))
        primary = members[0]
        support_count = len(members)
        quality = (
            -1.25 * float(placements[primary].rmsd_to_target)
            + (0.15 if placements[primary].is_specific else 0.0)
            + 0.9 * math.log2(max(1, support_count))
        )
        group_index = len(groups)
        group = _ConsensusGroup(
            members=members,
            residue_keys=frozenset(best_by_residue),
            primary=primary,
            quality=quality,
        )
        groups.append(group)
        for node in direct:
            unassigned.discard(node)
            node_to_group[node] = group_index
    return groups, node_to_group


@dataclass(frozen=True)
class _SearchState:
    groups: Tuple[int, ...]
    starts: Tuple[int, ...]
    used_residues: frozenset
    score: float


def _state_rank(state: _SearchState, groups: List[_ConsensusGroup]) -> Tuple:
    node_signature = tuple(groups[group].primary for group in state.groups)
    unique_bp = 3 + max(state.starts) - min(state.starts)
    return (-state.score, -unique_bp, node_signature)


def _beam_weighted_path(
    graph: OverlapGraph,
    placements: List[Placement],
    width: int,
) -> AssemblyPath:
    groups, node_to_group = _consensus_groups(graph, placements)
    if not groups:
        return AssemblyPath()

    transitions: Dict[int, Dict[int, CompatibilityEdge]] = defaultdict(dict)
    for edge in sorted(
        graph.edges.values(),
        key=lambda e: (e.source, e.target, e.offset, e.backbone_rmsd),
    ):
        if edge.offset not in (1, 2):
            continue
        source_group = node_to_group[edge.source]
        target_group = node_to_group[edge.target]
        if source_group == target_group:
            continue
        old = transitions[source_group].get(target_group)
        if old is None or (
            edge.backbone_rmsd, edge.offset, edge.source, edge.target
        ) < (
            old.backbone_rmsd, old.offset, old.source, old.target
        ):
            transitions[source_group][target_group] = edge

    beam = [
        _SearchState(
            groups=(group_index,),
            starts=(0,),
            used_residues=group.residue_keys,
            score=4.5 + group.quality,
        )
        for group_index, group in enumerate(groups)
    ]
    beam.sort(key=lambda state: _state_rank(state, groups))
    beam = beam[:width]
    best = beam[0]
    while beam:
        next_states: List[_SearchState] = []
        for state in beam:
            if _state_rank(state, groups) < _state_rank(best, groups):
                best = state
            source_group = state.groups[-1]
            for target_group, edge in sorted(
                transitions.get(source_group, {}).items(),
                key=lambda item: (item[1].offset, item[1].backbone_rmsd, item[0]),
            ):
                target = groups[target_group]
                if state.used_residues & target.residue_keys:
                    continue
                gap_penalty = 3.25 if edge.offset == 2 else 0.0
                edge_score = (
                    1.5 * edge.offset
                    - 2.0 * edge.backbone_rmsd
                    - gap_penalty
                    + target.quality
                )
                next_states.append(_SearchState(
                    groups=state.groups + (target_group,),
                    starts=state.starts + (state.starts[-1] + edge.offset,),
                    used_residues=state.used_residues | target.residue_keys,
                    score=state.score + edge_score,
                ))
        if not next_states:
            break
        deduplicated: Dict[Tuple[int, frozenset, int], _SearchState] = {}
        for state in next_states:
            key = (state.groups[-1], state.used_residues, state.starts[-1])
            old = deduplicated.get(key)
            if old is None or _state_rank(state, groups) < _state_rank(old, groups):
                deduplicated[key] = state
        beam = sorted(deduplicated.values(), key=lambda state: _state_rank(state, groups))[:width]

    selected: List[Tuple[int, int]] = []
    for group_index, start in zip(best.groups, best.starts):
        for node in groups[group_index].members:
            selected.append((start, node))
    selected.sort(key=lambda item: (item[0], _placement_key(placements[item[1]], item[1])))
    return AssemblyPath(
        [node for _, node in selected],
        [start for start, _ in selected],
        best.score,
    )


def longest_simple_path(adj: List[List[int]], placements: List[Placement]) -> List[int]:
    """Select a deterministic inference-only weighted assembly.

    The historical name is retained for API compatibility.  New overlap
    graphs optimize supported bp, local protein fit, zero-offset consensus,
    backbone overlap residual and a conservative +/-2 bridge penalty.
    """
    if isinstance(adj, OverlapGraph):
        width = max(1, int(os.environ.get("NTM_ASSEMBLY_BEAM_WIDTH", "512")))
        return _beam_weighted_path(adj, placements, width)

    edges: Dict[Tuple[int, int], CompatibilityEdge] = {}
    for source, neighbors in enumerate(adj):
        for target in sorted(neighbors):
            edges[(source, target)] = CompatibilityEdge(
                source, target, 1, 0.0, 0, (0, 0)
            )
    return _beam_weighted_path(OverlapGraph(adj, edges), placements, 512)


def merge_chain_atoms_in_order(
    ordered_nodes: List[int],
    placements: List[Placement],
    chains: List[str],
) -> Dict[str, List[DNAAtom]]:
    """Merge selected 3-mers by integer bp coordinates without duplicate atoms."""
    if len(chains) != 2:
        raise ValueError("3-mer duplex assembly requires exactly two DNA chains")
    valid_positions = [
        position for position, node in enumerate(ordered_nodes)
        if placements[node].has_complete_dna_fragment(chains)
        and has_valid_internal_dna_geometry(placements[node], chains)
    ]
    if not valid_positions:
        return {ch: [] for ch in chains}

    if isinstance(ordered_nodes, AssemblyPath) and (
        len(ordered_nodes.coordinates) == len(ordered_nodes)
    ):
        starts = list(ordered_nodes.coordinates)
    else:
        starts = list(range(len(ordered_nodes)))

    candidates: Dict[int, List[Tuple[Tuple, int, int]]] = defaultdict(list)
    for position in valid_positions:
        node = ordered_nodes[position]
        start = starts[position]
        placement = placements[node]
        for local_bp in range(3):
            priority = (
                abs(local_bp - 1),
                *_placement_key(placement, node),
            )
            candidates[start + local_bp].append((priority, node, local_bp))

    selected_bp = {
        coordinate: min(options, key=lambda item: item[0])[1:]
        for coordinate, options in candidates.items()
    }
    result: Dict[str, List[DNAAtom]] = {ch: [] for ch in chains}

    def collect_residue(
        placement: Placement, chain: str, local_bp: int,
    ) -> List[DNAAtom]:
        order = placement.chain_resseq_order()[chain]
        slot = local_bp if chain == chains[0] else 2 - local_bp
        source_resseq = order[slot]
        records = [
            atom for atom in placement.dna_atoms
            if atom[0] == chain and atom[1] == source_resseq
        ]
        unique: Dict[str, DNAAtom] = {}
        for atom in sorted(records, key=lambda rec: (_normalize_atom_name(rec[3]), rec[3])):
            atom_key = _normalize_atom_name(atom[3])
            if atom_key in unique:
                raise ValueError(
                    f"Duplicate DNA atom identity: chain={chain} resseq={source_resseq} atom={atom_key}"
                )
            unique[atom_key] = atom
        return list(unique.values())

    coordinates = sorted(selected_bp)
    for chain in chains:
        coordinate_order = coordinates if chain == chains[0] else list(reversed(coordinates))
        for output_resseq, coordinate in enumerate(coordinate_order, start=1):
            node, local_bp = selected_bp[coordinate]
            for _, _, resname, atom_name, xyz in collect_residue(
                placements[node], chain, local_bp
            ):
                result[chain].append(
                    (chain, output_resseq, resname, atom_name, xyz)
                )
    return result
