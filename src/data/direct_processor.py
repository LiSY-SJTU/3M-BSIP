
import logging
import os
from pathlib import Path

import torch
from Bio.PDB import PDBParser

from .extractors import ChainAnalyzer, ProteinFeatureExtractor

logger = logging.getLogger(__name__)

class DirectProcessor:

    def __init__(self, cache_dir='./direct_cache', core_length=8):
        self.cache_dir = cache_dir
        self.core_length = core_length
        os.makedirs(cache_dir, exist_ok=True)
    
    def process_pdb(self, pdb_path):
        
        pdb_path = Path(pdb_path)
        pdb_id = pdb_path.stem
        
        cache_path = Path(self.cache_dir) / f"{pdb_id}.pt"
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        
        try:
            logger.info(f"Processing file: {pdb_path}")
            
            abs_pdb_path = os.path.abspath(pdb_path)
            logger.info(f"Using absolute path: {abs_pdb_path}")
            
            structure = PDBParser(QUIET=True).get_structure(abs_pdb_path, abs_pdb_path)
            
            for model in structure:
                for chain in model:
                    for residue in list(chain):
                        for atom in list(residue):
                            if atom.element == 'H':
                                residue.detach_child(atom.id)
            
            chain_analyzer = ChainAnalyzer(structure, is_training=False)
            if not chain_analyzer.is_valid():
                logger.warning(f"Invalid chain structure: {pdb_path}")
                return None
            
            protein_extractor = ProteinFeatureExtractor(structure, chain_analyzer)
            if not protein_extractor.validate():
                logger.warning(f"Invalid protein features: {pdb_path}")
                return None
            
            protein_features = protein_extractor.extract_features()
            
            if 'surface' in protein_features:
                surface = protein_features['surface']
                
                batch_surface = {}
                for key, value in surface.items():
                    if isinstance(value, torch.Tensor):
                        batch_surface[key] = value.unsqueeze(0)
                
                if 'vertices' in batch_surface:
                    n_points = batch_surface['vertices'].size(1)
                    masks = torch.ones(1, n_points, dtype=torch.bool)
                    batch_surface['masks'] = masks
                    logger.info(f"Added batched surface mask with shape: {masks.shape}")
                
                protein_features['surface'] = batch_surface
            
            num_nodes = protein_features['structure'].num_nodes()
            contact_data = {
                'contact_tensor': torch.zeros(num_nodes, 4),
                'preference_tensor': torch.zeros(num_nodes, 4),
                'strength_tensor': torch.zeros(num_nodes)
            }
            
            dna_features = {'sequence': 'N' * self.core_length}
            
            result = {
                'protein_features': protein_features,
                'dna_features': dna_features,
                'contact_tensor': contact_data['contact_tensor'],
                'preference_tensor': contact_data['preference_tensor'],
                'strength_tensor': contact_data['strength_tensor'],
                'pdb_id': pdb_id
            }
            
            torch.save(result, cache_path)
            return result
            
        except Exception as e:
            logger.error(f"Error while processing {pdb_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None 
