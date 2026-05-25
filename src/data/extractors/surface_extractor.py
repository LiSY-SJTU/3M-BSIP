import logging
import random
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import NeighborSearch, Selection, Structure
from pykeops.torch import LazyTensor, Vi, Vj
from pykeops.torch.cluster import grid_cluster
from scipy.spatial import cKDTree
from skimage import measure

from ..constants import (ATOM_CHARGE_DICT, ATOM_RADIUS_DICT,
                         MAX_SURFACE_POINTS, MIN_SURFACE_POINTS,
                         POINTS_PER_ANGSTROM, SURFACE_RESOLUTION)
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

def ranges_slices_torch(batch):
    
    Ns = torch.bincount(batch)
    indices = Ns.cumsum(0)
    ranges = torch.cat((torch.zeros(1, dtype=indices.dtype, device=batch.device), indices))
    ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous()
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
    return ranges, slices

def diagonal_ranges_torch(batch_x=None, batch_y=None):
    
    if batch_x is None and batch_y is None:
        return None
    elif batch_y is None:
        batch_y = batch_x

    ranges_x, slices_x = ranges_slices_torch(batch_x)
    ranges_y, slices_y = ranges_slices_torch(batch_y)

    ranges_x = ranges_x.to(batch_x.device)
    ranges_y = ranges_y.to(batch_y.device)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x

class SurfaceExtractor(BaseExtractor):

    def __init__(self, structure: Structure, chain_analyzer):
        super().__init__(structure, chain_analyzer)

        self.distance_threshold = 1.05
        self.smoothness = 0.5
        self.optimization_nits = 5
        self.sup_sampling_ratio = 20
        self.variance_threshold = 0.5
        self.resolution = 1.0
        self.gradient_adjustment = 0.5
        
        self.max_points = MAX_SURFACE_POINTS
        self.min_points = MIN_SURFACE_POINTS
        
        self.neighbors_k = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.heavy_atoms = [atom for atom in Selection.unfold_entities(self.chain_analyzer.protein_chain, 'A')
                            if atom.element in ['C', 'N', 'O', 'S']]
        if not self.heavy_atoms:
             raise ValueError("No heavy atoms (C, N, O, S) found in the protein chain.")

        self.atom_coords_np = np.array([atom.get_coord() for atom in self.heavy_atoms])
        self.atom_coords = torch.tensor(self.atom_coords_np, dtype=torch.float32, device=self.device)

        default_radius = 1.7
        raw_radii = [ATOM_RADIUS_DICT.get(atom.element, default_radius) for atom in self.heavy_atoms]
        self.atom_radii_raw = torch.tensor(raw_radii, dtype=torch.float32, device=self.device)

        self.effective_atom_smoothness = (self.atom_radii_raw * self.smoothness).unsqueeze(-1).unsqueeze(-1)
        self.effective_atom_smoothness = self.effective_atom_smoothness.contiguous()

    def _compute_soft_distances_pykeops(self, points_y: torch.Tensor, batch_y: torch.Tensor = None) -> torch.Tensor:
        
        atoms_x = self.atom_coords
        smoothness_i = LazyTensor(self.effective_atom_smoothness)

        batch_x = torch.zeros(len(atoms_x), dtype=torch.long, device=self.device)

        if batch_y is None:
            batch_y = torch.zeros(len(points_y), dtype=torch.long, device=self.device)

        atoms_x = atoms_x.contiguous()
        points_y = points_y.contiguous()
        batch_x = batch_x.contiguous()
        batch_y = batch_y.contiguous()

        x_i = Vi(atoms_x)
        y_j = Vj(points_y)

        D_ij = ((x_i - y_j) ** 2).sum(-1)

        dist_ij = (D_ij + 1e-8).sqrt()
        weights_ij = (- dist_ij / smoothness_i ).exp()

        sum_weights_j = weights_ij.sum_reduction(axis=0)

        sum_weights_j = sum_weights_j + 1e-8

        weighted_smoothness_j = (smoothness_i * weights_ij).sum_reduction(axis=0)
        mean_smoothness_j = weighted_smoothness_j / sum_weights_j

        log_sum_exp_j = (- dist_ij / smoothness_i).logsumexp_reduction(axis=0)

        soft_dists = - mean_smoothness_j.view(-1) * log_sum_exp_j.view(-1)

        return soft_dists

    def _subsample_points(self, points: torch.Tensor, batch: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if batch is None:
            batch = torch.zeros(len(points), dtype=torch.long, device=points.device)

        points = points.contiguous()
        batch = batch.contiguous()

        scale_t = torch.tensor([self.resolution], dtype=points.dtype, device=points.device)

        labels = grid_cluster(points, scale_t).long()

        if labels.numel() == 0:
            logger.warning("Received empty labels in _subsample_points, returning empty tensors.")
            return torch.empty((0, points.shape[1]), dtype=points.dtype, device=points.device), \
                   torch.empty(0, dtype=batch.dtype, device=batch.device)

        C = labels.max() + 1

        points_1 = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
        D = points_1.shape[1]

        agg_points = torch.zeros((C, D), dtype=points.dtype, device=points.device)

        agg_points.scatter_add_(0, labels[:, None].repeat(1, D), points_1)

        counts = agg_points[:, -1:]
        subsampled_points = agg_points[:, :-1] / torch.clamp(counts, min=1e-8)

        unique_labels = torch.unique(labels, sorted=True)

        sorted_indices = torch.argsort(labels)
        sorted_labels = labels[sorted_indices]
        if sorted_labels.numel() == 0:
             first_occurrence_indices = torch.tensor([], dtype=torch.long, device=labels.device)
        else:
             label_change = torch.cat( (torch.tensor([True], device=labels.device), sorted_labels[1:] != sorted_labels[:-1]) )
             first_occurrence_indices = sorted_indices[label_change]

        if first_occurrence_indices.numel() > 0:
             subsampled_batch = batch[first_occurrence_indices]
        else:
             subsampled_batch = torch.tensor([], dtype=batch.dtype, device=batch.device)

        return subsampled_points.contiguous(), subsampled_batch.contiguous()

    def _generate_points_normals_optimized(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        atoms = self.atom_coords
        N, D = atoms.shape
        T = self.distance_threshold
        max_geometric_distance = 3.0

        num_initial_points = N * self.sup_sampling_ratio
        z = atoms.repeat_interleave(self.sup_sampling_ratio, dim=0) + \
            10 * T * torch.randn(num_initial_points, D, device=self.device, dtype=atoms.dtype)
        z = z.detach().contiguous()

        batch_z = torch.zeros(num_initial_points, dtype=torch.long, device=self.device)

        

        z.requires_grad_(True)
        for it in range(self.optimization_nits):
            if z.grad is not None:
                z.grad.zero_()

            dists = self._compute_soft_distances_pykeops(z, batch_z)
            loss = ((dists - T) ** 2).sum()
            
            if not loss.requires_grad:
                 logger.warning(f"Loss does not require grad at iteration {it+1}. Stopping optimization.")
                 z.requires_grad_(False)
                 break

            try:
                 g = torch.autograd.grad(loss, z, create_graph=False)[0]
                 if g is None:
                     logger.warning(f"Gradient is None at iteration {it+1}. Stopping optimization.")
                     z.requires_grad_(False)
                     break
                 z.data -= 0.5 * g
            except RuntimeError as e:
                 logger.error(f"RuntimeError during gradient computation at iter {it+1}: {e}")
                 z.requires_grad_(False)
                 break

        z.requires_grad_(False)
        z = z.contiguous()

        

        if len(z) > 0:
            try:
                atoms_filt = atoms.contiguous().to(z.device)
                points_filt = z.contiguous()

                x_i = Vi(atoms_filt)
                y_j = Vj(points_filt)

                D_ij = ((x_i - y_j) ** 2).sum(-1)

                min_sq_dist_j = D_ij.min_reduction(axis=0)

                min_dist_j = min_sq_dist_j.sqrt().view(-1)
                mask_geometric = min_dist_j < max_geometric_distance

                original_len = len(z)
                z = z[mask_geometric]
                batch_z = batch_z[mask_geometric]
                

            except Exception as e_filt:
                 logger.error(f"Error during PyKeOps geometric filtering: {e_filt}", exc_info=True)
                 logger.warning("Skipping PyKeOps geometric filtering due to error.")

        if len(z) == 0:
             logger.warning("No points remained after geometric filtering. Returning empty tensors.")
             return torch.empty((0, 3), device=self.device), torch.empty((0, 3), device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        dists = self._compute_soft_distances_pykeops(z, batch_z)
        margin = (dists - T).abs()
        mask_margin = margin < self.variance_threshold * T

        z_margin_filtered = z[mask_margin]
        batch_z_margin_filtered = batch_z[mask_margin]
        zz = z_margin_filtered.detach().clone()

        mask_inside = torch.ones(len(zz), dtype=torch.bool, device=self.device)
        if len(zz) > 0:
            zz.requires_grad_(True)
            for it in range(self.optimization_nits):
                if zz.grad is not None: zz.grad.zero_()
                dists_zz = self._compute_soft_distances_pykeops(zz, batch_z_margin_filtered)
                loss_zz = (1.0 * dists_zz).sum()
                if not loss_zz.requires_grad: logger.warning(f"Loss (outward push) does not require grad at iter {it+1}. Stopping."); zz.requires_grad_(False); break
                try:
                    g_zz = torch.autograd.grad(loss_zz, zz, create_graph=False)[0]
                    if g_zz is None: logger.warning(f"Gradient (outward push) is None at iter {it+1}. Stopping."); zz.requires_grad_(False); break
                    normals_zz = F.normalize(g_zz, p=2, dim=-1)
                    zz.data += 1.0 * T * normals_zz
                except RuntimeError as e: logger.error(f"RuntimeError (outward push) at iter {it+1}: {e}"); zz.requires_grad_(False); break

            zz.requires_grad_(False)
            dists_final_zz = self._compute_soft_distances_pykeops(zz, batch_z_margin_filtered)
            mask_inside = dists_final_zz > 1.5 * T

        z_filtered = z_margin_filtered[mask_inside]
        batch_z_filtered = batch_z_margin_filtered[mask_inside]

        if len(z_filtered) == 0:
             logger.warning("No points remained after filtering. Returning empty tensors.")
             return torch.empty((0, 3), device=self.device), torch.empty((0, 3), device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        points, batch_points = self._subsample_points(z_filtered, batch_z_filtered)

        if len(points) < self.min_points:
             logger.warning(f"Number of subsampled points ({len(points)}) is less than min_points ({self.min_points}). Consider adjusting resolution or other parameters.")

        points = points.detach().clone()
        points.requires_grad_(True)
        dists_final = self._compute_soft_distances_pykeops(points, batch_points)
        loss_final = (1.0 * dists_final).sum()

        if not loss_final.requires_grad:
            logger.error("Cannot compute final normals, loss does not require grad.")
            normals = torch.zeros_like(points)
        else:
            try:
                 g_final = torch.autograd.grad(loss_final, points, create_graph=False)[0]
                 if g_final is None:
                     logger.error("Final gradient for normals is None.")
                     normals = torch.zeros_like(points)
                 else:
                    normals = F.normalize(g_final, p=2, dim=-1)
            except RuntimeError as e:
                 logger.error(f"RuntimeError during final normal computation: {e}")
                 normals = torch.zeros_like(points)

        points.requires_grad_(False)
        normals = normals.detach()

        points = points - self.gradient_adjustment * normals

        return points.contiguous(), normals.contiguous(), batch_points.contiguous()

    def _get_atom_types_and_residue_indices(self, surface_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if len(surface_points) == 0:
            return torch.empty((0, 4), dtype=torch.float32, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        atom_coords_np = self.atom_coords_np
        tree = cKDTree(atom_coords_np)

        surface_points_np = surface_points.cpu().numpy()
        _, indices = tree.query(surface_points_np, k=1)

        atom_types_list = []
        residue_indices_list = []
        atom_element_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

        for i in range(len(surface_points)):
            nearest_atom_index = indices[i]
            nearest_atom = self.heavy_atoms[nearest_atom_index]

            atom_type = torch.zeros(4, dtype=torch.float32)
            element_idx = atom_element_map.get(nearest_atom.element)
            if element_idx is not None:
                 atom_type[element_idx] = 1.0
            atom_types_list.append(atom_type)

            residue = nearest_atom.get_parent()
            residue_indices_list.append(residue.get_id()[1])

        atom_types = torch.stack(atom_types_list).to(self.device)
        residue_indices = torch.tensor(residue_indices_list, dtype=torch.long, device=self.device)

        return atom_types, residue_indices

    def extract_features(self) -> Dict:

        logger.info("Starting surface extraction using optimization method...")

        if not hasattr(self, 'atom_coords') or self.atom_coords is None:
             self.atom_coords = torch.tensor(self.atom_coords_np, dtype=torch.float32, device=self.device)

        surface_points, normals, batch_indices = self._generate_points_normals_optimized()

        if len(surface_points) == 0:
            logger.warning("Surface generation yielded no points. Returning empty features.")
            return {
                'vertices': torch.empty((0, 3), dtype=torch.float32),
                'normals': torch.empty((0, 3), dtype=torch.float32),
                'charges': torch.empty(0, dtype=torch.float32),
                'atom_types': torch.empty((0, 4), dtype=torch.float32),
                'neighbors': torch.empty((0, self.neighbors_k), dtype=torch.long),
                'residue_indices': torch.empty(0, dtype=torch.long),
                'distance_features': torch.empty((0, self.neighbors_k), dtype=torch.float32),
                'angle_features': torch.empty((0, self.neighbors_k), dtype=torch.float32),
                'curvature': torch.empty(0, dtype=torch.float32),
                'local_frames': torch.empty((0, 4, 3), dtype=torch.float32),
                'relative_positions': torch.empty((0, self.neighbors_k, 3), dtype=torch.float32),
            }

        atom_types, residue_indices = self._get_atom_types_and_residue_indices(surface_points)

        surface_points_np = surface_points.cpu().numpy()
        normals_np = normals.cpu().numpy()

        charges_np = self._compute_charges(surface_points_np, self.heavy_atoms)
        charges = torch.tensor(charges_np, dtype=torch.float32, device=self.device)

        neighbors_np = self._compute_neighbors(surface_points_np, k=self.neighbors_k)
        neighbors = torch.tensor(neighbors_np, dtype=torch.long, device=self.device)

        invariant_features_np = self._compute_local_invariant_features(surface_points_np, normals_np, neighbors_np)

        local_frames_np = self._compute_relative_frames(surface_points_np, normals_np)
        relative_positions_np = self._compute_relative_positions(surface_points_np, local_frames_np, neighbors_np)

        features = {
            'vertices': surface_points,
            'normals': normals,
            'charges': charges,
            'atom_types': atom_types,
            'neighbors': neighbors,
            'residue_indices': residue_indices,
            'distance_features': torch.tensor(invariant_features_np['distance_features'], dtype=torch.float32, device=self.device),
            'angle_features': torch.tensor(invariant_features_np['angle_features'], dtype=torch.float32, device=self.device),
            'curvature': torch.tensor(invariant_features_np['curvature'], dtype=torch.float32, device=self.device),
            'local_frames': torch.tensor(local_frames_np, dtype=torch.float32, device=self.device),
            'relative_positions': torch.tensor(relative_positions_np, dtype=torch.float32, device=self.device)
        }

        logger.info(f"Extracted surface features with keys: {list(features.keys())}")
        for key, value in features.items():
             if isinstance(value, torch.Tensor):
                 logger.info(f"  {key}: {value.shape} on {value.device}")
        else:
                 logger.info(f"  {key}: type {type(value)}")

        return features

    def _compute_charges(self, points: np.ndarray, atoms: List) -> np.ndarray:
        
        charges = np.zeros(len(points))
        atom_coords = np.array([atom.get_coord() for atom in atoms])
        
        atom_charges = []
        for atom in atoms:
            residue = atom.get_parent()
            charge_key = (atom.element, residue.resname)
            charge = ATOM_CHARGE_DICT.get(charge_key, 
                    ATOM_CHARGE_DICT.get((atom.element, 'ALL'), 0.0))
            atom_charges.append(charge)
        atom_charges = np.array(atom_charges)

        if len(atom_coords) == 0: return charges
        
        tree = cKDTree(atom_coords)
        k_query = min(5, len(atom_coords))
        if k_query == 0: return charges

        dists, indices = tree.query(points, k=k_query)

        if k_query == 1:
             dists = dists[:, np.newaxis]
             indices = indices[:, np.newaxis]

        weights = 1.0 / (dists + 1e-10)
        sum_weights = weights.sum(axis=1, keepdims=True)
        weights = weights / np.where(sum_weights == 0, 1, sum_weights)
        
        charges = np.sum(weights * atom_charges[indices], axis=1)
        
        return charges

    def _compute_neighbors(self, points: np.ndarray, k: int = None) -> np.ndarray:
        
        if k is None:
            k = self.neighbors_k
        if len(points) <= k:
             logger.warning(f"Number of points ({len(points)}) is less than or equal to k ({k}). Cannot compute k distinct neighbors.")
             num_points = len(points)
             if num_points == 0:
                 return np.empty((0, k), dtype=np.int64)
             indices = np.arange(num_points)
             repeats = (k // num_points) + 1
             neighbors_padded = np.tile(indices, repeats)[:k]
             all_neighbors = np.tile(neighbors_padded, (num_points, 1))
             for i in range(num_points):
                 mask = all_neighbors[i] == i
                 if np.any(mask):
                     available_neighbors = np.setdiff1d(indices, [i], assume_unique=True)
                     if len(available_neighbors) > 0:
                         all_neighbors[i, mask] = np.random.choice(available_neighbors, size=np.sum(mask))
             return all_neighbors

        tree = cKDTree(points)
        _, indices = tree.query(points, k=k+1)
        return indices[:, 1:]
    
    def _calculate_local_curvature(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        
        logger.info("Calculating local curvature using normal variations...")
        if len(points) == 0: return np.array([])

        points = np.asarray(points)
        normals = np.asarray(normals)

        search_radius = 3.0
        min_neighbors_for_curvature = 5

        tree = cKDTree(points)
        curvatures = np.zeros(len(points))

        for i in range(len(points)):
            neighbors_indices = tree.query_ball_point(points[i], search_radius)
            neighbors_indices = [idx for idx in neighbors_indices if idx != i]

            if len(neighbors_indices) < min_neighbors_for_curvature:
                curvatures[i] = 0.0
                continue
                
            neighbor_normals = normals[neighbors_indices]
            center_normal = normals[i]

            dot_products = np.dot(neighbor_normals, center_normal)
            dot_products = np.clip(dot_products, -1.0, 1.0)
            angular_diff = np.arccos(dot_products)
            curvatures[i] = np.mean(angular_diff)

        if len(curvatures) > 0:
             min_curv, max_curv = np.min(curvatures), np.max(curvatures)
             if max_curv > min_curv:
                 curvatures = (curvatures - min_curv) / (max_curv - min_curv)
             else:
                 curvatures = np.zeros_like(curvatures)

        logger.info(f"Curvature calculation complete, shape: {curvatures.shape}")
        return curvatures

    def _compute_local_invariant_features(self, points: np.ndarray, normals: np.ndarray, neighbors: np.ndarray) -> Dict:
        
        logger.info("Computing local invariant features...")
        num_points = len(points)
        if num_points == 0:
            return {'distance_features': np.empty((0, self.neighbors_k)),
                    'angle_features': np.empty((0, self.neighbors_k)),
                    'curvature': np.empty(0)}

        k = neighbors.shape[1]
        local_features = {}
        
        neighbor_points = points[neighbors.reshape(-1)].reshape(num_points, k, 3)
        diffs = neighbor_points - points[:, np.newaxis, :]
        neighbor_dists = np.linalg.norm(diffs, axis=2)
        local_features['distance_features'] = neighbor_dists
        
        angle_features = np.zeros((num_points, k))
        center_normals = normals[:, np.newaxis, :]
        neighbor_normals = normals[neighbors.reshape(-1)].reshape(num_points, k, 3)

        dot_products = np.sum(center_normals * neighbor_normals, axis=2)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angle_features = np.arccos(dot_products)
        local_features['angle_features'] = angle_features
        
        local_features['curvature'] = self._calculate_local_curvature(points, normals)
        
        logger.info("Local invariant features computed.")
        return local_features

    def _compute_relative_frames(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        
        logger.info("Computing local frames...")
        num_points = len(points)
        local_frames = np.zeros((num_points, 4, 3))

        basis1 = normals

        cand_basis2 = np.zeros_like(normals)
        use_x_axis = np.abs(basis1[:, 0]) < np.abs(basis1[:, 1])
        cand_basis2[use_x_axis, 0] = 1.0
        cand_basis2[~use_x_axis, 1] = 1.0

        dot_prod = np.sum(cand_basis2 * basis1, axis=1, keepdims=True)
        basis2 = cand_basis2 - dot_prod * basis1

        norms_basis2 = np.linalg.norm(basis2, axis=1, keepdims=True)
        basis2 = basis2 / (norms_basis2 + 1e-10)

        basis3 = np.cross(basis1, basis2, axisa=1, axisb=1)

        local_frames[:, 0] = points
        local_frames[:, 1] = basis1
        local_frames[:, 2] = basis2
        local_frames[:, 3] = basis3

        logger.info("Local frames computed.")
        return local_frames

    def _compute_relative_positions(self, points: np.ndarray, local_frames: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        
        logger.info("Computing relative positions...")
        n_points = len(points)
        if n_points == 0:
             return np.empty((0, self.neighbors_k, 3))
        k = neighbors.shape[1]
        
        relative_positions = np.zeros((n_points, k, 3))
        
        origin = local_frames[:, 0]
        basis1 = local_frames[:, 1]
        basis2 = local_frames[:, 2]
        basis3 = local_frames[:, 3]

        neighbor_coords = points[neighbors.reshape(-1)].reshape(n_points, k, 3)

        relative_vec_global = neighbor_coords - origin[:, np.newaxis, :]

        x_local = np.einsum('ni,nki->nk', basis2, relative_vec_global)
        y_local = np.einsum('ni,nki->nk', basis3, relative_vec_global)
        z_local = np.einsum('ni,nki->nk', basis1, relative_vec_global)

        relative_positions = np.stack([x_local, y_local, z_local], axis=-1)

        logger.info("Relative positions computed.")
        return relative_positions