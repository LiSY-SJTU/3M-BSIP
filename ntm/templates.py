from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import os
import numpy as np
from Bio.PDB import PDBParser
from .pdb_utils import extract_reference_atoms
from .assembly import dna_atoms_internal_clash_count


@dataclass
class Template:
    template_id: str
    aa_type: str
    nt_type: str
    ref_names: List[str]
    ref_coords: np.ndarray
    dna_atoms: List[Tuple[str, int, str, str, np.ndarray]]


def parse_template_pdb(
    pdb_path: str,
) -> Tuple[
    Optional[List[Dict]], Optional[List[Tuple[str, int, str, str, np.ndarray]]]
]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tpl", pdb_path)
    model = next(structure.get_models())
    protein_chains = []
    dna_chains = {}
    for chain in model:
        if chain.id in ("A",):
            protein_chains.append(chain)
        elif chain.id in ("B", "C"):
            dna_chains[chain.id] = chain

    if len(protein_chains) != 1 or set(dna_chains) != {"B", "C"}:
        return None, None

    a_chain = protein_chains[0]
    residues = [r for r in a_chain if r.id[0] == " "]
    if len(residues) != 5:
        return None, None
    block: List[Dict] = []
    for r in residues:
        atoms = {}
        for atom in r:
            if atom.element == "H":
                continue
            atoms[atom.name.strip()] = atom.coord.copy()
        block.append({
            "chain": "A",
            "resseq": int(r.id[1]),
            "resname": r.get_resname().upper(),
            "atoms": atoms,
        })

    dna_atoms: List[Tuple[str, int, str, str, np.ndarray]] = []
    for chain_id in ("B", "C"):
        chain = dna_chains[chain_id]
        residues = [residue for residue in chain if residue.id[0] == " "]

        if len(residues) != 3:
            return None, None

        resseqs = [int(residue.id[1]) for residue in residues]
        if len(set(resseqs)) != 3:
            return None, None

        for residue in residues:
            atoms = [atom for atom in residue if atom.element != "H"]
            if not atoms:
                return None, None
            resseq = int(residue.id[1])
            resname = residue.get_resname()
            for atom in atoms:
                dna_atoms.append((chain.id, resseq, resname, atom.name.strip(), atom.coord.copy()))

    if dna_atoms_internal_clash_count(dna_atoms, ["B", "C"]) > 0:
        return None, None

    return (
        block,
        dna_atoms,
    )


def load_templates(root_dir: str) -> Iterable[Template]:
    for aa_type in sorted(os.listdir(root_dir)):
        aa_dir = os.path.join(root_dir, aa_type)
        if not os.path.isdir(aa_dir):
            continue
        for nt_type in sorted(os.listdir(aa_dir)):
            nt_dir = os.path.join(aa_dir, nt_type)
            if not os.path.isdir(nt_dir):
                continue
            for fname in sorted(os.listdir(nt_dir)):
                if not fname.lower().endswith(".pdb"):
                    continue
                pdb_path = os.path.join(nt_dir, fname)
                template_id = os.path.splitext(fname)[0]

                result = parse_template_pdb(pdb_path)
                if result[0] is None or result[1] is None:
                    continue

                block, dna_atoms = result
                ref_names, ref_coords = extract_reference_atoms(block, center_index=2)

                yield Template(
                    template_id=template_id,
                    aa_type=aa_type.upper(),
                    nt_type=nt_type,
                    ref_names=ref_names,
                    ref_coords=ref_coords,
                    dna_atoms=dna_atoms,
                )
