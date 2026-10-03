from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree


def build_protein_kdtree(protein_atoms: List[Tuple[str, int, str, np.ndarray]]) -> cKDTree:
    coords = np.vstack([a[3] for a in protein_atoms])
    return cKDTree(coords)


def has_protein_clash(
    dna_atoms: List[Tuple[str, int, str, str, np.ndarray]],
    protein_tree: cKDTree,
    clash_distance: float,
) -> bool:
    dna_coords = np.vstack([a[4] for a in dna_atoms])
    dists, _ = protein_tree.query(dna_coords, k=1)
    return bool(np.any(dists < clash_distance))
