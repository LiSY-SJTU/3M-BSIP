from typing import List, Optional, Tuple

import numpy as np
from Bio.PDB import Chain, Residue, Vector
from Bio.PDB.vectors import calc_dihedral


def calculate_backbone_angles(residue: Residue) -> Tuple[float, float]:
    
    phi = psi = 0.0
    try:
        chain = residue.get_parent()
        res_id = residue.id[1]
        
        prev_res = next((r for r in chain if r.id[1] == res_id - 1), None)
        next_res = next((r for r in chain if r.id[1] == res_id + 1), None)
        
        if prev_res and all(atom in prev_res for atom in ['C']) and \
           all(atom in residue for atom in ['N', 'CA', 'C']):
            phi = calc_dihedral(
                prev_res['C'].get_vector(),
                residue['N'].get_vector(),
                residue['CA'].get_vector(),
                residue['C'].get_vector()
            )
        
        if next_res and all(atom in residue for atom in ['N', 'CA', 'C']) and \
           all(atom in next_res for atom in ['N']):
            psi = calc_dihedral(
                residue['N'].get_vector(),
                residue['CA'].get_vector(),
                residue['C'].get_vector(),
                next_res['N'].get_vector()
            )
    except Exception:
        pass
    
    return phi, psi

def calculate_local_frame(residue: Residue) -> np.ndarray:
    
    try:
        n = residue['N'].get_coord()
        ca = residue['CA'].get_coord()
        c = residue['C'].get_coord()
        
        forward = normalize(c - ca)
        backward = normalize(n - ca)
        up = normalize(np.cross(forward, backward))
        right = normalize(np.cross(forward, up))
        
        return np.stack([forward, right, up])
        
    except Exception:
        return np.zeros((3, 3))

def calculate_min_distance(res1: Residue, res2: Residue) -> float:
    
    min_dist = float('inf')
    for atom1 in res1:
        for atom2 in res2:
            dist = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
            min_dist = min(min_dist, dist)
    return min_dist

def calculate_relative_orientation(coord1: np.ndarray, 
                                coord2: np.ndarray, 
                                local_frame: np.ndarray) -> np.ndarray:
    
    direction = (coord2 - coord1)
    distance = np.linalg.norm(direction)
    if distance > 1e-6:
        direction = direction / distance
    return np.dot(local_frame, direction)

def calculate_centroid(coords: np.ndarray) -> np.ndarray:
    
    return np.mean(coords, axis=0)

def normalize(v: np.ndarray) -> np.ndarray:
    
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm

def get_neighbor_indices(coords: np.ndarray, 
                        cutoff: float, 
                        exclude_self: bool = True) -> List[Tuple[int, int]]:
    
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = list(tree.query_pairs(cutoff))
    if not exclude_self:
        pairs.extend([(i, i) for i in range(len(coords))])
    return sorted(pairs) 