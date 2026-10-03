from typing import Dict, List, Tuple
import numpy as np
from Bio.PDB import PDBParser
from .sidechain import BACKBONE_ORDER, get_sidechain_order


def load_pdb_atoms(pdb_path: str) -> List[Tuple[str, int, str, np.ndarray]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_path)
    atoms: List[Tuple[str, int, str, np.ndarray]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resseq = int(residue.id[1])
                for atom in residue:
                    if atom.element == "H":
                        continue
                    atoms.append((chain.id, resseq, atom.name.strip(), atom.coord.copy()))
        break
    return atoms


def extract_reference_atoms(
    residues: List[Dict], center_index: int
) -> Tuple[List[str], np.ndarray]:
    """Return reference atoms keyed by position relative to the center residue.

    The template library retains residue numbers from its source PDB entries,
    while a target structure has unrelated numbering.  Using a relative offset
    makes the keys comparable while preserving the special all-atom treatment
    of the central residue.
    """
    names: List[str] = []
    coords: List[np.ndarray] = []

    if center_index >= len(residues):
        raise IndexError("Center residue index is outside the residue block")

    for offset, res in enumerate(residues):
        is_center = (offset == center_index)
        if is_center:
            order = BACKBONE_ORDER + get_sidechain_order(res["resname"])
        else:
            order = BACKBONE_ORDER
        atom_dict: Dict[str, np.ndarray] = res["atoms"]
        for an in order:
            if an in atom_dict:
                coord = atom_dict[an]
                relative_position = offset - center_index
                names.append(f"{relative_position}:{an}")
                coords.append(coord)
            else:
                raise ValueError(
                    f"Residue {res['resseq']} is missing required atom {an}"
                )

    if len(coords) == 0:
        raise ValueError("无法提取任何参考原子")

    return names, np.vstack(coords)


def collect_residue_blocks(
    pdb_path: str,
    chain_id: str,
    center_resseq: int,
) -> List[Dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]
    resseqs = [r.id[1] for r in chain if r.id[0] == " "]
    idx = resseqs.index(center_resseq)
    window = resseqs[max(0, idx - 2) : idx + 3]

    if len(window) != 5:
        raise ValueError(
            f"Residue {center_resseq} does not have a complete five-residue window"
        )

    block: List[Dict] = []
    for resseq in window:
        try:
            residue = chain[(" ", resseq, " ")]
            resname = residue.get_resname().upper()
            atoms: Dict[str, np.ndarray] = {}
            for atom in residue:
                if atom.element == "H":
                    continue
                atoms[atom.name.strip()] = atom.coord.copy()
            block.append({"chain": chain_id, "resseq": int(resseq), "resname": resname, "atoms": atoms})
        except KeyError as exc:
            raise KeyError(f"Residue {resseq} is missing from chain {chain_id}") from exc

    if len(block) != 5:
        raise ValueError(f"Residue block for {center_resseq} is incomplete")

    return block
