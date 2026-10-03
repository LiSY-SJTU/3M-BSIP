from typing import Dict, List, Optional, Tuple
import os
import csv
import numpy as np
from .templates import Template, load_templates
from .pdb_utils import collect_residue_blocks, extract_reference_atoms, load_pdb_atoms
from .geometry import kabsch, apply_transform, rmsd
from .clash import build_protein_kdtree, has_protein_clash
from .assembly import Placement, build_overlap_graph, longest_simple_path, merge_chain_atoms_in_order


def index_templates(root_dir: str) -> Dict[Tuple[str, str], List[Template]]:
    index: Dict[Tuple[str, str], List[Template]] = {}
    templates_found = 0
    for tpl in load_templates(root_dir):
        key = (tpl.aa_type, tpl.nt_type)
        index.setdefault(key, []).append(tpl)
        templates_found += 1

    print(f"模板索引完成：找到 {templates_found} 个有效模板")

    return index


def read_interface_csv(path: str) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["chain_id"], int(row["resseq"])) )
    return rows


def read_pred_nt_csv(path: str) -> Dict[Tuple[str, int], str]:
    mapping: Dict[Tuple[str, int], str] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[(row["chain_id"], int(row["resseq"]))] = row["pred_nt"].upper()
    return mapping


def select_templates_for(templates_index: Dict[Tuple[str, str], List[Template]], aa: str, nt: str) -> List[Template]:
    return templates_index.get((aa, nt), [])


def generate_best_placements(
    target_pdb: str,
    templates_root: str,
    interface_list: List[Tuple[str, int]],
    pred_nt_map: Dict[Tuple[str, int], str],
    rmsd_threshold: float,
    clash_distance: float,
    templates_index: Optional[Dict[Tuple[str, str], List[Template]]] = None,
) -> Tuple[List[Placement], Dict[Tuple[str, int], Dict[str, int]]]:
    if templates_index is None:
        templates_index = index_templates(templates_root)
    protein_atoms = load_pdb_atoms(target_pdb)
    protein_tree = build_protein_kdtree(protein_atoms)

    placements: List[Placement] = []
    residue_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

    for (chain_id, resseq) in interface_list:
        block = collect_residue_blocks(target_pdb, chain_id, resseq)
        center_index = next(
            index for index, residue in enumerate(block)
            if residue["resseq"] == resseq
        )
        aa_type = block[center_index]["resname"]
        ref_names_tgt, ref_coords_tgt = extract_reference_atoms(block, center_index)

        nt = pred_nt_map[(chain_id, resseq)]
        tpls = select_templates_for(templates_index, aa_type, nt)

        key = (chain_id, resseq)
        if key not in residue_stats:
            residue_stats[key] = {"matched": 0, "clash_pass": 0, "aa": aa_type}

        for tpl in tpls:
            name_to_idx_tpl = {n: i for i, n in enumerate(tpl.ref_names)}
            name_to_idx_tgt = {n: i for i, n in enumerate(ref_names_tgt)}
            common_names = [n for n in tpl.ref_names if n in name_to_idx_tgt]

            if len(common_names) < 10:
                continue

            idx_tpl = [name_to_idx_tpl[n] for n in common_names]
            idx_tgt = [name_to_idx_tgt[n] for n in common_names]

            P_sub = tpl.ref_coords[idx_tpl, :]
            Q_sub = ref_coords_tgt[idx_tgt, :]

            R, t = kabsch(P_sub, Q_sub)
            fitted_sub = apply_transform(P_sub, R, t)
            d = rmsd(fitted_sub, Q_sub)
            if d > rmsd_threshold:
                continue

            residue_stats[key]["matched"] += 1

            transformed_atoms = []
            for (c, r, resname, a, xyz) in tpl.dna_atoms:
                new_xyz = (R @ xyz.reshape(3, 1)).reshape(3,) + t
                transformed_atoms.append((c, r, resname, a, new_xyz))

            if has_protein_clash(transformed_atoms, protein_tree, clash_distance):
                continue

            residue_stats[key]["clash_pass"] += 1

            pl = Placement(
                protein_chain_id=chain_id,
                protein_resseq=resseq,
                aa_type=aa_type,
                template_id=tpl.template_id,
                rmsd_to_target=d,
                is_specific=(tpl.nt_type != "Generic"),
                dna_atoms=transformed_atoms,
            )
            placements.append(pl)

    return placements, residue_stats


def assemble_and_export(
    placements: List[Placement],
    overlap_threshold: float,
    out_pdb: str,
    out_csv: str,
    residue_stats: Dict[Tuple[str, int], Dict[str, int]],
) -> None:
    if not placements:
        raise ValueError("No valid ntM placements were generated")

    print(f"装配候选数量：{len(placements)}")
    chains = ["B", "C"]
    adj = build_overlap_graph(placements, chains, threshold=overlap_threshold)
    order_idx = longest_simple_path(adj, placements)

    if order_idx and hasattr(order_idx, "coordinates"):
        assembly_coordinates = list(order_idx.coordinates)
        unique_bp = len({
            start + local_bp
            for start in assembly_coordinates
            for local_bp in range(3)
        })
        print(
            "装配选择："
            f"placements={len(order_idx)}，unique_bp={unique_bp}，"
            f"score={order_idx.score:.3f}，starts={assembly_coordinates}"
        )

    if not order_idx:
        raise ValueError("No geometrically compatible ntM assembly was found")
    merged = merge_chain_atoms_in_order(order_idx, placements, chains)

    lines: List[str] = []
    serial = 1
    for ch in chains:
        res_to_atoms: Dict[int, List[Tuple[str, int, str, str, np.ndarray]]] = {}
        atom_keys = set()
        for rec in merged[ch]:
            atom_key = (ch, rec[1], rec[3])
            if atom_key in atom_keys:
                raise ValueError(
                    "Duplicate DNA atom identity before PDB export: "
                    f"chain={ch} resseq={rec[1]} atom={rec[3]}"
                )
            atom_keys.add(atom_key)
            res_to_atoms.setdefault(rec[1], []).append(rec)
        new_resseq = 1
        for resseq in sorted(res_to_atoms.keys()):
            atoms = res_to_atoms[resseq]
            resname = atoms[0][2]
            for (_, _, _, atom_name, xyz) in atoms:
                x, y, z = xyz.tolist()
                line = (
                    f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {ch}{new_resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           "
                )
                lines.append(line)
                serial += 1
            new_resseq += 1
    with open(out_pdb, "w") as f:
        for line in lines:
            f.write(line + "\n")
        f.write("END\n")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chain_id", "resseq", "aa_type", "template_id", "rmsd"])
        chosen = [placements[i] for i in order_idx]
        for pl in chosen:
            writer.writerow([pl.protein_chain_id, pl.protein_resseq, pl.aa_type, pl.template_id, f"{pl.rmsd_to_target:.3f}"])

    resid_to_aa = {(pl.protein_chain_id, pl.protein_resseq): pl.aa_type for pl in placements}
    for key, stat in residue_stats.items():
        aa = stat.get("aa")
        if aa is not None:
            resid_to_aa[key] = aa
    selected_count: Dict[Tuple[str, int], int] = {}
    for pl in chosen:
        k = (pl.protein_chain_id, pl.protein_resseq)
        selected_count[k] = selected_count.get(k, 0) + 1
    for key in sorted(residue_stats.keys(), key=lambda x: (x[0], x[1])):
        ch, resseq = key
        stats = residue_stats[key]
        aa = resid_to_aa.get(key, "?")
        selected = 1 if selected_count.get(key, 0) > 0 else 0
        print(f"统计 {ch}:{resseq} ({aa})：匹配={stats.get('matched', 0)}，无碰撞={stats.get('clash_pass', 0)}，选中={selected}")


def run_pipeline(
    target_pdb: str,
    templates_root: str,
    interface_csv: str,
    pred_nt_csv: str,
    outdir: str,
    rmsd_threshold: float = 2.0,
    clash_distance: float = 2.0,
    overlap_threshold: float = 0.8,
) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    interface_list = read_interface_csv(interface_csv)
    pred_map = read_pred_nt_csv(pred_nt_csv)
    placements, residue_stats = generate_best_placements(
        target_pdb,
        templates_root,
        interface_list,
        pred_map,
        rmsd_threshold,
        clash_distance,
    )
    out_pdb = os.path.join(outdir, "predicted_dna.pdb")
    out_csv = os.path.join(outdir, "placements.csv")
    assemble_and_export(placements, overlap_threshold, out_pdb, out_csv, residue_stats)
    return out_pdb, out_csv
