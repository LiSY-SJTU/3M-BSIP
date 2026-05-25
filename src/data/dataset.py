import logging
import multiprocessing
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import dgl
import numpy as np
import torch
import tqdm
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .extractors import (ChainAnalyzer, DNAFeatureExtractor,
                         ProteinFeatureExtractor)

logger = logging.getLogger(__name__)

def ensure_cache_dir(cache_dir):
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logger.info(f"Created feature cache directory: {cache_dir}")

class TFDNADataset(Dataset):

    def __init__(self, 
                 pdb_paths: List[str], 
                 cache_dir: str = './feature_cache', 
                 core_length: int = 8, 
                 num_workers: int = 60,
                 is_training: bool = True,
                 build_eval_labels: bool = False):
        
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        ensure_cache_dir(cache_dir)
        self.pdb_paths = pdb_paths
        self.core_length = core_length
        self.cache_dir = cache_dir
        self.is_training = is_training
        self.build_eval_labels = build_eval_labels
        
        self._process_uncached_files(num_workers)
        
        self.valid_indices = self._collect_valid_indices()
        self.original_order = {idx: i for i, idx in enumerate(self.valid_indices)}
        if is_training:
            self._sort_by_protein_length()
        logger.info(f"Found {len(self.valid_indices)} valid complexes")
    
    def _sort_by_protein_length(self):
        
        self.protein_lengths = []
        for idx in self.valid_indices:
            pdb_path = self.pdb_paths[idx]
            cache_path = self._get_cache_path(pdb_path)
            data = torch.load(cache_path)
            self.protein_lengths.append(data['contact_tensor'].size(0))
        
        sorted_indices = sorted(
            range(len(self.valid_indices)),
            key=lambda i: self.protein_lengths[i],
            reverse=True
        )
        
        self.valid_indices = [self.valid_indices[i] for i in sorted_indices]
        self.protein_lengths = [self.protein_lengths[i] for i in sorted_indices]
    
    def get_original_index(self, dataset_idx):
        
        return self.original_order[self.valid_indices[dataset_idx]]
    
    def _process_uncached_files(self, num_workers: int):
        
        uncached_files = [
            pdb_path for pdb_path in self.pdb_paths
            if not os.path.exists(self._get_cache_path(pdb_path))
        ]
        
        if uncached_files:
            logger.info(f"Processing {len(uncached_files)} uncached files with {num_workers} workers...")
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(num_workers) as pool:
                list(tqdm.tqdm(
                    pool.imap(self._process_and_cache, uncached_files),
                    total=len(uncached_files)
                ))
    
    def _collect_valid_indices(self) -> List[int]:
        
        valid_indices = []
        for idx, pdb_path in enumerate(self.pdb_paths):
            cache_path = self._get_cache_path(pdb_path)
            if os.path.exists(cache_path):
                data = torch.load(cache_path)
                if data is not None:
                    if self.is_training:
                        dna_seq = data['dna_features']['sequence']
                        dna_len = len(dna_seq)
                        if dna_len == self.core_length:
                            valid_indices.append(idx)
                        else:
                            logger.warning(f"Skipping {pdb_path} due to invalid DNA sequence length: {dna_len}, expected: {self.core_length}, sequence: {dna_seq}")
                    else:
                        valid_indices.append(idx)
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        pdb_idx = self.valid_indices[idx]
        pdb_path = self.pdb_paths[pdb_idx]
        cache_path = self._get_cache_path(pdb_path)
        
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        else:
            data = self.process_complex(pdb_path)
            if data is not None:
                torch.save(data, cache_path)
            return data
    
    def _get_cache_path(self, pdb_path: str) -> str:
        
        if isinstance(pdb_path, Path):
            pdb_path = str(pdb_path)
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
        split_name = os.path.basename(os.path.dirname(os.path.abspath(pdb_path)))
        return os.path.join(self.cache_dir, split_name, f"{pdb_id}.pt")
    
    def _process_and_cache(self, pdb_path: str) -> bool:
        
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            data = self.process_complex(pdb_path)
            if data is not None:
                cache_path = self._get_cache_path(pdb_path)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(data, cache_path)
                return True
        except Exception as e:
            logger.error(f"Error processing {pdb_path}: {str(e)}")
        return False
    
    def process_complex(self, pdb_path: str) -> Dict:
        
        try:
            pdb_path = os.path.abspath(pdb_path)
            structure = PDBParser(QUIET=True).get_structure(pdb_path, pdb_path)
            
            for model in structure:
                for chain in model:
                    for residue in list(chain):
                        for atom in list(residue):
                            if atom.element == 'H':
                                residue.detach_child(atom.id)
            
            chain_analyzer = ChainAnalyzer(structure, is_training=self.is_training or self.build_eval_labels)
            if not chain_analyzer.is_valid():
                logger.warning(f"Invalid chain structure in {pdb_path}")
                return None
                
            protein_extractor = ProteinFeatureExtractor(structure, chain_analyzer)
            if not protein_extractor.validate():
                logger.warning(f"Invalid protein features in {pdb_path}")
                return None
            protein_features = protein_extractor.extract_features()
            
            contact_data = {}
            if self.is_training or self.build_eval_labels:
                dna_extractor = DNAFeatureExtractor(structure, chain_analyzer)
                if not dna_extractor.validate():
                    logger.warning(f"Invalid DNA features in {pdb_path}")
                    return None
                    
                core_region = dna_extractor.extract_core_region(self.core_length)
                dna_features = dna_extractor.extract_features(core_region,self.core_length)
                contact_data = dna_extractor.build_contact_tensor(
                    core_region, 
                    protein_features['structure'].num_nodes()
                )
            else:
                dna_features = {'sequence': 'N' * self.core_length}
                num_nodes = protein_features['structure'].num_nodes()
                contact_data = {
                    'contact_tensor': torch.zeros(num_nodes, 4),
                    'preference_tensor': torch.zeros(num_nodes, 4),
                    'strength_tensor': torch.zeros(num_nodes)
                }
            
            return {
                'protein_features': protein_features,
                'dna_features': dna_features,
                'contact_tensor': contact_data['contact_tensor'],
                'preference_tensor': contact_data['preference_tensor'],
                'strength_tensor': contact_data['strength_tensor'],
                'pdb_id': os.path.splitext(os.path.basename(pdb_path))[0]
            }
            
        except Exception as e:
            logger.error(f"Error processing {pdb_path}: {str(e)}")
            return None

def move_structure_to_cpu(item):
    if isinstance(item, torch.Tensor):
        return item.cpu()
    elif isinstance(item, dict):
        return {k: move_structure_to_cpu(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_structure_to_cpu(elem) for elem in item]
    elif isinstance(item, tuple):
        return tuple(move_structure_to_cpu(elem) for elem in item)
    elif isinstance(item, dgl.DGLGraph):
        new_graph = item.clone()
        for key, feat in new_graph.ndata.items():
            if isinstance(feat, torch.Tensor):
                new_graph.ndata[key] = feat.cpu()
        for key, feat in new_graph.edata.items():
            if isinstance(feat, torch.Tensor):
                new_graph.edata[key] = feat.cpu()
        return new_graph
    else:
        return item

def collate_fn(batch: List[Dict], core_length: int) -> Dict:
    
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("Empty batch received in collate_fn after filtering Nones.")

    aa_to_id = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21, '-': 0
    }
    
    dna_to_id = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4, '-': 5}
    
    def seq_to_ids(seq: str, mapping: Dict[str, int]) -> List[int]:
        return [mapping.get(aa, mapping['-']) for aa in seq]
    
    max_protein_length = max(item['contact_tensor'].size(0) for item in batch)
    
    def pad_tensor(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
        
        pad_size = max_len - tensor.size(0)
        if pad_size > 0:
            padding = torch.zeros(pad_size, *tensor.size()[1:], 
                                dtype=tensor.dtype, 
                                device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
        return tensor
    
    protein_lengths = torch.LongTensor([
        item['contact_tensor'].size(0) for item in batch
    ])
    
    try:
        protein_sequences = []
        combined_sequences = []
        max_seq_len = max(len(item['protein_features']['sequence']['sequence']) 
                         for item in batch)
        
        for item in batch:
            seq = item['protein_features']['sequence']['sequence']
            seq_ids = seq_to_ids(seq, aa_to_id)
            padded_seq = seq_ids + [0] * (max_seq_len - len(seq_ids))
            protein_sequences.append(padded_seq)
            
            combined_seq = item['protein_features']['sequence']['combined_seq']
            combined_sequences.append(combined_seq)

        
        protein_sequences = torch.LongTensor(protein_sequences)
        
        for i, item in enumerate(batch):
            item['protein_features']['sequence'].update({
                'sequence_ids': protein_sequences[i],
                'combined_seq': combined_sequences[i]
            })
        
        dna_sequences = []
        dna_seq_len = core_length
        
        for item in batch:
            seq = item['dna_features']['sequence']
            seq_ids = seq_to_ids(seq, dna_to_id)
            if len(seq_ids) != dna_seq_len:
                raise ValueError(f"Invalid DNA sequence length: {len(seq_ids)}, expected {dna_seq_len}")
            dna_seq = torch.tensor(seq_ids, dtype=torch.long)
            dna_sequences.append(dna_seq)
        
        dna_sequences = torch.stack(dna_sequences)
        
        surface_features = {
            'vertices': [],
            'normals': [],
            'charges': [],
            'atom_types': [],
            'neighbors': [],
            'residue_indices': [],
            'distance_features': [],
            'angle_features': [],
            'curvature': [],
            'local_frames': [],
            'relative_positions': [],
            'masks': []
        }
        max_surface_points = max(item['protein_features']['surface']['vertices'].size(0) for item in batch)
        
        for item in batch:
            surface = item['protein_features']['surface']
            n_points = surface['vertices'].size(0)
            
            def pad_tensor(tensor, max_len):
                pad_size = max_len - tensor.size(0)
                if pad_size > 0:
                    if tensor.dim() == 1:
                        padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
                    else:
                        padding = torch.zeros(pad_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                    return torch.cat([tensor, padding], dim=0)
                return tensor
            
            surface_features['vertices'].append(pad_tensor(surface['vertices'], max_surface_points))
            surface_features['normals'].append(pad_tensor(surface['normals'], max_surface_points))
            surface_features['charges'].append(pad_tensor(surface['charges'], max_surface_points))
            surface_features['atom_types'].append(pad_tensor(surface['atom_types'], max_surface_points))
            surface_features['neighbors'].append(pad_tensor(surface['neighbors'], max_surface_points))
            surface_features['residue_indices'].append(pad_tensor(surface['residue_indices'], max_surface_points))
            
            surface_features['distance_features'].append(pad_tensor(surface['distance_features'], max_surface_points))
            surface_features['angle_features'].append(pad_tensor(surface['angle_features'], max_surface_points))
            surface_features['curvature'].append(pad_tensor(surface['curvature'], max_surface_points))
            surface_features['local_frames'].append(pad_tensor(surface['local_frames'], max_surface_points))
            surface_features['relative_positions'].append(pad_tensor(surface['relative_positions'], max_surface_points))
            
            mask = torch.zeros(max_surface_points, dtype=torch.bool)
            mask[:n_points] = True
            surface_features['masks'].append(mask)
        
    except Exception as e:
        logger.error(f"Error in sequence processing: {str(e)}")
        logger.error(f"Protein features structure: {batch[0]['protein_features']}")
        logger.error(f"DNA features structure: {batch[0]['dna_features']}")
        raise
    
    batched_data = {
        'protein_features': {
            'structure': dgl.batch([item['protein_features']['structure'] for item in batch]),
            'surface': {
                'vertices': torch.stack(surface_features['vertices']),
                'normals': torch.stack(surface_features['normals']),
                'charges': torch.stack(surface_features['charges']),
                'atom_types': torch.stack(surface_features['atom_types']),
                'neighbors': torch.stack(surface_features['neighbors']),
                'residue_indices': torch.stack(surface_features['residue_indices']),
                'masks': torch.stack(surface_features['masks']),
                'distance_features': torch.stack(surface_features['distance_features']),
                'angle_features': torch.stack(surface_features['angle_features']),
                'curvature': torch.stack(surface_features['curvature']),
                'local_frames': torch.stack(surface_features['local_frames']),
                'relative_positions': torch.stack(surface_features['relative_positions'])
            },
            'sequence': {
                'sequence_ids': protein_sequences,
                'combined_seq': combined_sequences
            }
        },
        'dna_features': {
            'sequence': dna_sequences
        },
        'contact_tensor': torch.stack([
            pad_tensor(item['contact_tensor'], max_protein_length) 
            for item in batch
        ]),
        'preference_tensor': torch.stack([
            pad_tensor(item['preference_tensor'], max_protein_length) 
            for item in batch
        ]),
        'strength_tensor': torch.stack([
            pad_tensor(item['strength_tensor'], max_protein_length) 
            for item in batch
        ]),
        'protein_lengths': torch.LongTensor([
            item['contact_tensor'].size(0) for item in batch
        ])
    }

    batched_data_cpu = move_structure_to_cpu(batched_data)

    return batched_data_cpu
