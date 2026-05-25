

import logging
import os
import subprocess
import tempfile
from shutil import which
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def get_foldseek_path() -> str:
    
    foldseek_path = os.getenv('FOLDSEEK_PATH')
    if foldseek_path and os.path.exists(foldseek_path):
        return foldseek_path
        
    foldseek_exec = which('foldseek')
    if foldseek_exec:
        return foldseek_exec
        
    default_paths = [
        'bin/foldseek',
        os.path.expanduser('~/mambaforge/envs/TF_DNA/bin/foldseek')
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(
        "Foldseek executable not found. Please install Foldseek or set FOLDSEEK_PATH."
    )

def get_struc_seq(pdb_path: str, chain_ids: List[str]) -> Dict[str, Tuple[str, str, str]]:
    
    try:
        foldseek_path = get_foldseek_path()
        logger.debug(f"Using Foldseek at: {foldseek_path}")
        
        with tempfile.NamedTemporaryFile(suffix='.a3m') as temp_file:
            cmd = [
                foldseek_path,
                "structureto3didescriptor",
                os.path.abspath(pdb_path),
                temp_file.name
            ]
            
            result = subprocess.run(
                cmd, 
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Foldseek command failed: {' '.join(cmd)}")
                logger.error(f"Stderr: {result.stderr}")
                raise RuntimeError(f"Foldseek failed with return code {result.returncode}")
            
            results = {}
            with open(temp_file.name, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                        
                    pdb_chain_id = parts[0]

                    sequence = parts[1]
                    structure_seq = parts[2].lower()
                        
                    if len(sequence) == len(structure_seq):
                        combined_seq = ''.join(
                                f"{aa}{struct}" 
                                for aa, struct in zip(sequence, structure_seq)
                        )
                        results[chain_ids[0]] = (sequence, structure_seq, combined_seq)
                    else:
                        logger.warning(
                                f"Sequence length mismatch for chain {chain_ids[0]}: "
                                f"seq={len(sequence)}, struct={len(structure_seq)}"
                        )
            
            if not results:
                logger.error(f"No valid sequences found in {pdb_path}")
                raise ValueError("No valid sequences extracted")
                
            return results
            
    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}")
        raise 