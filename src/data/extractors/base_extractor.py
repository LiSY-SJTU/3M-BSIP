from typing import Any, Dict

from Bio.PDB import Structure


class BaseExtractor:

    def __init__(self, structure: Structure, chain_analyzer: 'ChainAnalyzer'):
        
        self.structure = structure
        self.chain_analyzer = chain_analyzer
    
    def extract_features(self) -> Dict[str, Any]:
        
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def validate(self) -> bool:
        
        return True 