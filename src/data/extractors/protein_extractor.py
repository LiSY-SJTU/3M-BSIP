import logging
import os
from typing import Dict, List

import dgl
import numpy as np
import torch
from Bio.PDB import *
from scipy.spatial import cKDTree

from ..constants import AMINO_ACID_MAPPING
from ..utils.foldseek_util import get_struc_seq
from ..utils.geometry_util import (calculate_backbone_angles,
                                   calculate_local_frame, normalize)
from .base_extractor import BaseExtractor
from .surface_extractor import SurfaceExtractor

logger = logging.getLogger(__name__)

class ProteinFeatureExtractor(BaseExtractor):

    def __init__(self, structure, chain_analyzer, neighbor_cutoff: float = 6.0):
        
        super().__init__(structure, chain_analyzer)
        self.neighbor_cutoff = neighbor_cutoff
    
    def extract_features(self) -> Dict:
        
        protein_chain = self.chain_analyzer.protein_chain
        
        return {
            'structure': self.build_structure_graph(protein_chain),
            'surface': self.extract_surface_features(),
            'sequence': self.extract_sequence_features(protein_chain)
        }
    
    def build_structure_graph(self, chain) -> dgl.DGLGraph:
        
        residues = []
        coords = []
        for residue in chain:
            if 'CA' in residue: 
                residues.append(residue)
                coords.append(residue['CA'].get_coord())
        
        coords = np.array(coords)
        
        kdtree = cKDTree(coords)
        pairs = list(kdtree.query_pairs(self.neighbor_cutoff))
        
        node_features = [self._get_residue_features(res) for res in residues]
        edge_features = self._build_edge_features(pairs, coords, residues)
        
        g = dgl.graph((
            [p[0] for p in pairs] + [p[1] for p in pairs],
            [p[1] for p in pairs] + [p[0] for p in pairs]
        ), num_nodes=len(residues))
        
        g.ndata['feat'] = torch.tensor(np.array(node_features))
        g.edata['feat'] = torch.tensor(np.array(edge_features))
        
        return g
    
    def extract_surface_features(self) -> Dict:
        
        surface_extractor = SurfaceExtractor(self.structure, self.chain_analyzer)
        return surface_extractor.extract_features()
    
    def extract_sequence_features(self, chain) -> Dict:
        
        pdb_path = os.path.abspath(self.structure.id)
        chain_id = chain.id
        
        struc_seqs = get_struc_seq(pdb_path, [chain_id])
        if chain_id not in struc_seqs:
            logger.error(f"No sequence found for chain {chain_id} in {pdb_path}")
            raise ValueError(f"No sequence found for chain {chain_id}")
        
        sequence, structure_seq, combined_seq = struc_seqs[chain_id]
        
        if not sequence:
            logger.error(f"Empty sequence extracted for chain {chain_id} in {pdb_path}")
            raise ValueError("Empty sequence extracted")

        return {
            'sequence': sequence,
            'structure': structure_seq,
            'combined_seq': combined_seq
        }
    
    def validate(self) -> bool:
        
        try:
            chain = self.chain_analyzer.protein_chain
            if not chain:
                self.last_validation_error = "No protein chain found"
                return False
            
            valid_residues = [res for res in chain if 'CA' in res]
            if len(valid_residues) < 10:
                self.last_validation_error = f"Too few valid residues: {len(valid_residues)}"
                return False
            
            try:
                seq_features = self.extract_sequence_features(chain)
                if not seq_features['sequence']:
                    self.last_validation_error = "Empty sequence features"
                    return False
            except Exception as e:
                self.last_validation_error = f"Sequence extraction failed: {str(e)}"
                return False
            
            return True
            
        except Exception as e:
            self.last_validation_error = f"Validation error: {str(e)}"
            return False
    
    def _get_residue_features(self, residue) -> np.ndarray:
        
        phi, psi = calculate_backbone_angles(residue)
        local_frame = calculate_local_frame(residue)
        
        return np.concatenate([
            [phi, psi],
            local_frame.flatten()
        ])
    
    def _build_edge_features(self, pairs, coords, residues) -> List[np.ndarray]:
        
        edge_features = []
        for i, j in pairs:
            forward = self._get_edge_features(
                coords[i], coords[j], 
                residues[i], residues[j]
            )
            edge_features.append(forward)
            
            backward = self._get_edge_features(
                coords[j], coords[i], 
                residues[j], residues[i]
            )
            edge_features.append(backward)
        
        return edge_features
    
    def _get_edge_features(self, coord1, coord2, res1, res2) -> np.ndarray:
        
        distance = np.linalg.norm(coord1 - coord2)
        
        direction = (coord2 - coord1) / (distance + 1e-6)
        local_frame = calculate_local_frame(res1)
        relative_direction = np.dot(local_frame, direction)
        
        seq_dist = abs(res1.get_id()[1] - res2.get_id()[1])
        
        return np.concatenate([
            [distance],
            relative_direction,
            [seq_dist]
        ])