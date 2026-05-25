from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio.PDB import Residue

from ..constants import (BASE_ATOMS, CONTACT_DISTANCE_CUTOFF,
                         INTERACTION_DISCOUNT, NUCLEOTIDE_MAPPING)
from .base_extractor import BaseExtractor


class DNAFeatureExtractor(BaseExtractor):

    def __init__(self, structure, chain_analyzer):
        super().__init__(structure, chain_analyzer)
        self.distance_cutoff = CONTACT_DISTANCE_CUTOFF
    
    def extract_features(self, core_region: Dict,core_length: int) -> Dict:
        
        dna_chain = None
        for chain in self.chain_analyzer.dna_chains:
            if chain.id == core_region['chain_id']:
                dna_chain = chain
                break
            
        if dna_chain is None:
            raise ValueError(f"Cannot find DNA chain with ID {core_region['chain_id']}")
        
        sequence = ""
        coordinates = []
        
        for res in dna_chain:
            res_id = res.id[1]
            if core_region['start'] <= res_id < core_region['end']:
                sequence += self._get_nucleotide_name(res.resname)
                if "C1'" in res:
                    coordinates.append(res["C1'"].get_coord())
        
        if len(sequence) != core_length:
            sequence += 'N' * (core_length - len(sequence))
        return {
            'sequence': sequence,
            'coordinates': np.array(coordinates)
        }
    
    def extract_core_region(self, core_length: int) -> Dict:
        
        all_atom_contacts = {}
        
        all_dna_positions = {}
        for dna_chain in self.chain_analyzer.dna_chains:
            chain_id = dna_chain.id
            DNA_chain_contacts = self._calculate_chain_atom_contacts(
                self.chain_analyzer.protein_chain, 
                dna_chain
            )
            all_atom_contacts.update(DNA_chain_contacts)
            
            if chain_id not in all_dna_positions:
                all_dna_positions[chain_id] = []
            
            for res in dna_chain:
                if res.resname.strip() in NUCLEOTIDE_MAPPING:
                    all_dna_positions[chain_id].append(res.id[1])
        
        best_core_region = None
        max_contacts = -1
        
        for chain_id, positions in all_dna_positions.items():
            chain_contacts = {
                k.split(":")[-1]: v for k, v in all_atom_contacts.items()
                if k.startswith(f"{chain_id}:")
            }
            
            core_region = self._find_densest_atom_contacts(chain_contacts, positions, core_length)
            
            if core_region['contact_count'] > max_contacts:
                max_contacts = core_region['contact_count']
                best_core_region = core_region
                best_core_region['chain_id'] = chain_id
        
        if best_core_region is None:
            raise ValueError("No valid core contact region found")
            
        return best_core_region
    
    def _calculate_chain_atom_contacts(self, protein_chain, dna_chain) -> Dict[str, int]:
        
        nucleotide_atom_counts = {}
        chain_id = dna_chain.id
        
        for nt_res in dna_chain:
            if nt_res.resname.strip() not in NUCLEOTIDE_MAPPING:
                continue
            
            nt_position = nt_res.id[1]
            nt_key = f"{chain_id}:{nt_position}"
            nucleotide_atom_counts[nt_key] = 0
            
            for nt_atom in nt_res:
                if nt_atom.name.startswith('H'):
                    continue
                    
                for aa_res in protein_chain:
                    for aa_atom in aa_res:
                        if aa_atom.name.startswith('H'):
                            continue
                        
                        distance = np.linalg.norm(
                            nt_atom.get_coord() - aa_atom.get_coord()
                        )
                        
                        if distance <= self.distance_cutoff:
                            nucleotide_atom_counts[nt_key] += 1
                
        return nucleotide_atom_counts
    
    def build_contact_tensor(self, core_region: Dict, num_protein_nodes: int) -> Dict[str, torch.Tensor]:
        
        contact_tensor = torch.zeros(num_protein_nodes, 4)
        preference_tensor = torch.zeros(num_protein_nodes, 4)
        strength_tensor = torch.zeros(num_protein_nodes)
        
        protein_chain = self.chain_analyzer.protein_chain
        protein_residues = list(protein_chain)
        
        core_chain_id = core_region['chain_id']
        core_dna_chain = None
        for chain in self.chain_analyzer.dna_chains:
            if chain.id == core_chain_id:
                core_dna_chain = chain
                break

        for protein_idx, protein_res in enumerate(protein_residues):
            if 'CA' not in protein_res:
                continue
            
            closest_dna_res = None
            min_distance = float('inf')
            
            for dna_res in core_dna_chain:
                res_id = dna_res.id[1]
                if not (core_region['start'] <= res_id < core_region['end']):
                    continue
                
                dist = self.calculate_chemically_aware_min_distance(protein_res, dna_res)
                
                if dist < min_distance:
                    min_distance = dist
                    closest_dna_res = dna_res
            
            if closest_dna_res is not None and min_distance <= 4.5:
                probs_dict = self._calculate_nucleotide_probabilities(protein_res, closest_dna_res)
                contact_tensor[protein_idx] = torch.tensor(probs_dict['final_probs'])
                preference_tensor[protein_idx] = torch.tensor(probs_dict['preference_probs'])
                strength_tensor[protein_idx] = probs_dict['contact_strength']
        
        return {
            'contact_tensor': contact_tensor,
            'preference_tensor': preference_tensor,
            'strength_tensor': strength_tensor
        }
    
    def _find_densest_atom_contacts(self, 
                                   atom_contacts: Dict, 
                                   positions: List[int],
                                   core_length: int) -> Dict:
        
        if not atom_contacts:
            return {'start': 0, 'end': core_length, 'contact_count': 0}
        
        sorted_positions = sorted(set(positions))
        
        best_start = sorted_positions[0]
        best_end = sorted_positions[min(core_length-1, len(sorted_positions)-1)]
        max_atom_contacts = 0
        
        for pos in range(best_start, best_end + 1):
            if str(pos) in atom_contacts:
                max_atom_contacts += atom_contacts[str(pos)]

        for i in range(len(sorted_positions) - core_length + 1):
            window_start = sorted_positions[i]
            window_end = sorted_positions[i + core_length - 1]
            
            window_atom_contacts = 0
            for pos in range(window_start, window_end + 1):
                if str(pos) in atom_contacts:
                    window_atom_contacts += atom_contacts[str(pos)]
            
            if window_atom_contacts > max_atom_contacts:
                max_atom_contacts = window_atom_contacts
                best_start = window_start
                best_end = window_end
        
        return {
            'start': best_start,
            'end': best_end + 1,
            'contact_count': max_atom_contacts
        }
    
    def _calculate_nucleotide_probabilities(self, 
                                          aa_residue: Residue, 
                                          dna_residue: Residue) -> Dict[str, List[float]]:
        
        min_dist = self.calculate_chemically_aware_min_distance(aa_residue, dna_residue)
        
        contact_strength = np.exp(-0.5 * (min_dist / 3.0)**2)
        
        base_dist = min(self._calculate_base_distance(aa_residue, dna_residue), 4.5)
        real_nt_idx = self._get_nucleotide_index(dna_residue.resname)
        
        logits = np.zeros(4)
        logits[real_nt_idx] = 12.0 / (1.0 + np.exp(base_dist/2.0))
        for i in range(4):
            if i != real_nt_idx:
                logits[i] = 2.0 / (1.0 + np.exp((4.0 - base_dist)/2.0))
        
        preference_probs = torch.softmax(torch.tensor(logits), dim=0).tolist()
        
        final_probs = [contact_strength * p for p in preference_probs]
        
        return {
            'contact_strength': contact_strength,
            'preference_probs': preference_probs,
            'final_probs': final_probs
        }
    
    @staticmethod
    def _find_dna_residue(dna_chain, position: int) -> Residue:
        for res in dna_chain:
            if res.id[1] == position:
                return res
        return None
    
    @staticmethod
    def _get_nucleotide_name(resname: str) -> str:
        return NUCLEOTIDE_MAPPING.get(resname.strip(), ('N', -1))[0]
    
    @staticmethod
    def _get_nucleotide_index(resname: str) -> int:
        return NUCLEOTIDE_MAPPING.get(resname.strip(), ('N', -1))[1]
    
    def _calculate_base_distance(self, aa_residue: Residue, dna_residue: Residue) -> float:
        min_dist = float('inf')
        for aa_atom in aa_residue:
            if not self._is_sidechain_heavy_atom(aa_atom.name):
                continue
                
            for base_atom in BASE_ATOMS:
                if base_atom in dna_residue:
                    dist = np.linalg.norm(
                        aa_atom.get_coord() - dna_residue[base_atom].get_coord()
                    )
                    min_dist = min(min_dist, dist)
        
        if min_dist == float('inf'):
            return 99
            
        return min_dist
    
    @staticmethod
    def _is_sidechain_heavy_atom(atom_name: str) -> bool:
        
        if atom_name.startswith('H'):
            return False
            
        if atom_name in {'N', 'CA', 'C', 'O', 'OXT'}:
            return False
            
        return True

    def _get_atom_types(self, atom):
        
        res_name = atom.get_parent().resname.strip()
        atom_name = atom.name
        
        possible_types = []
        
        nonpolar_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TYR', 'TRP'}
        if res_name in nonpolar_residues and atom_name in ['CB', 'CG', 'CG1', 'CG2', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3','CZ', 'CZ2', 'CZ3', 'CH2']:
            possible_types.append('HYDROPHOBIC')
        polar_residues = {'SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR', 'HIS'}
        if res_name in polar_residues and atom_name in ['OG', 'OG1', 'OH']:
            possible_types.extend(['H_DONOR', 'H_ACCEPTOR'])
        if res_name in polar_residues and atom_name in ['OD1', 'OE1']:
            possible_types.append('H_ACCEPTOR')
        if res_name in polar_residues and atom_name in ['ND2', 'NE2']:
            possible_types.append('H_DONOR')

        positive_residues = {'ARG', 'LYS', 'HIS'}
        if res_name in positive_residues and atom_name in ['NH1', 'NH2', 'NZ','ND1', 'NE2']:
            possible_types.extend(['POSITIVE', 'H_DONOR'])
        
        negative_residues = {'ASP', 'GLU'}
        if res_name in negative_residues and atom_name in ['OD1', 'OD2', 'OE1', 'OE2']:
            possible_types.extend(['NEGATIVE', 'H_ACCEPTOR'])
        
        if atom_name == 'N':
            possible_types.append('H_DONOR')
        if atom_name == 'O':
            possible_types.append('H_ACCEPTOR')
        
        if res_name == 'DA' :
            if atom_name in ['N1', 'N3', 'N7']:
                possible_types.append('H_ACCEPTOR')
            if atom_name in ['N6']:
                possible_types.append('H_DONOR')
            if atom_name in ['OP1', 'OP2']:
                possible_types.extend(['H_ACCEPTOR','NEGATIVE'])

        if res_name == 'DT' :
            if atom_name in ['O2', 'O4']:
                possible_types.append('H_ACCEPTOR')
            if atom_name in ['N3']:
                possible_types.append('H_DONOR')
            if atom_name in ['OP1', 'OP2']:
                possible_types.extend(['H_ACCEPTOR','NEGATIVE'])
            if atom_name in ['C7']:
                possible_types.append('HYDROPHOBIC')
        
        if res_name == 'DC' :
            if atom_name in ['O2','N3']:
                possible_types.append('H_ACCEPTOR')
            if atom_name in ['N4']:
                possible_types.append('H_DONOR')
            if atom_name in ['OP1', 'OP2']:
                possible_types.extend(['H_ACCEPTOR','NEGATIVE'])

        if res_name == 'DG' :
            if atom_name in ['O6','N3','N7']:
                possible_types.append('H_ACCEPTOR')
            if atom_name in ['N1','N2']:
                possible_types.append('H_DONOR')
            if atom_name in ['OP1', 'OP2']:
                possible_types.extend(['H_ACCEPTOR','NEGATIVE'])

        if not possible_types:
            possible_types.append('OTHER')
        
        return possible_types

    def calculate_chemically_aware_distance(self, atom1, atom2, interaction_discount):
        
        types1 = self._get_atom_types(atom1)
        types2 = self._get_atom_types(atom2)
        
        geo_distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
        
        min_discount = 1.0
        for type1 in types1:
            for type2 in types2:
                discount = interaction_discount.get((type1, type2), 1.0)
                if discount < min_discount:
                    min_discount = discount

        return geo_distance * min_discount

    def calculate_chemically_aware_min_distance(self, residue1, residue2, distance_threshold=4.5):
        
        min_distance = 99
        
        for atom1 in residue1:
            if atom1.name.startswith('H'): 
                continue
            
            for atom2 in residue2:
                if atom2.name.startswith('H'): 
                    continue
                
                abs_distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                
                if abs_distance <= distance_threshold:
                    chem_distance = self.calculate_chemically_aware_distance(
                        atom1, atom2, INTERACTION_DISCOUNT
                    )
                    
                    if chem_distance < min_distance:
                        min_distance = chem_distance
        
        return min_distance