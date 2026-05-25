import logging
from typing import List, Optional, Union

from Bio.PDB import Chain, Structure

from ..constants import DNA_RATIO_THRESHOLD, MIN_DNA_LENGTH, NUCLEOTIDE_MAPPING
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

class ChainAnalyzer(BaseExtractor):

    def __init__(self, structure: Structure, is_training: bool = True):
        
        super().__init__(structure, None)
        self._protein_chain: Optional[Chain.Chain] = None
        self._dna_chains: Optional[List[Chain.Chain]] = None
        self.is_training = is_training
        self._analyze_chains()
    
    def extract_features(self) -> dict:
        return {
            'protein_chain': self.protein_chain.id if self.protein_chain else None,
            'dna_chains': [chain.id for chain in self.dna_chains] if self.dna_chains else None
        }
    
    def _analyze_chains(self):
        for chain in self.structure[0]:
            if self.is_training and self.is_dna_chain(chain):
                if self._dna_chains is None:
                    self._dna_chains = []
                self._dna_chains.append(chain)
            else:
                if self._protein_chain is None:
                    self._protein_chain = chain
    
    @property
    def protein_chain(self) -> Chain.Chain:
        
        if self._protein_chain is None:
            raise ValueError("No protein chain found in structure")
        return self._protein_chain
    
    @property
    def dna_chains(self) -> List[Chain.Chain]:
        
        if self._dna_chains is None:
            raise ValueError("No DNA chains found in structure")
        return self._dna_chains
    
    def validate(self) -> bool:
        
        if not self.is_valid():
            return False
            
        if self.is_training and not self.check_dna_length():
            return False
            
        if not self._check_protein_chain():
            return False
            
        return True
    
    def is_valid(self) -> bool:
        
        if self.is_training:
            return (self._protein_chain is not None and 
                    self._dna_chains is not None and 
                    len(self._dna_chains) > 0)
        else:
            return self._protein_chain is not None
    
    @staticmethod
    def is_dna_chain(chain: Chain.Chain) -> bool:
        
        residues = set(res.resname.strip() for res in chain)
        has_dna = bool(residues & set(NUCLEOTIDE_MAPPING.keys()))
        
        if has_dna:
            total_residues = len([res for res in chain])
            dna_residues = len([res for res in chain 
                               if res.resname.strip() in NUCLEOTIDE_MAPPING])
            dna_ratio = dna_residues / total_residues if total_residues > 0 else 0
            return dna_ratio > DNA_RATIO_THRESHOLD
        
        return False
    
    def check_dna_length(self) -> bool:
        
        if not self._dna_chains:
            return False
            
        total_length = sum(
            sum(1 for res in chain if res.resname.strip() in NUCLEOTIDE_MAPPING)
            for chain in self._dna_chains
        )
        return total_length >= MIN_DNA_LENGTH
    
    def _check_protein_chain(self) -> bool:
        
        if not self._protein_chain:
            return False
            
        valid_residues = [res for res in self._protein_chain if 'CA' in res]
        return len(valid_residues) >= 10