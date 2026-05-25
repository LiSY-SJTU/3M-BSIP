

import logging
import os
from typing import Any, Dict, Tuple

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..modules.attention import MultiModalFusion
from ..modules.gnn import StructureEncoderGVP
from ..modules.surface_gnn import SurfaceGNNEncoder
from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)

class HierarchicalSurfaceEncoder(nn.Module):

    def __init__(self, hidden_dim: int, num_neighbors: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.local_encoder = dMaSIF(
            config=dMaSIFConfig(
                hidden_dim=hidden_dim,
                num_neighbors=num_neighbors,
                curvature_scales=[1.0, 2.0, 3.0, 5.0],
                patch_radius=9.0,
                num_heads=8,
                feat_drop=0.1,
                attn_drop=0.1
            )
        )

        self.patch_encoder = nn.ModuleList([
            dMaSIFConv(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=8,
                num_neighbors=num_neighbors
            ) for _ in range(3)
        ])

        self.residue_pooling = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.cross_residue_conv = dMaSIFConvEquivariant(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=8,
            num_neighbors=16
        )

        self.residue_norm = nn.LayerNorm(hidden_dim)
        self.surface_norm = nn.LayerNorm(hidden_dim)

    def forward(self, surface_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N = surface_data['vertices'].shape[:2]

        local_feat = self.local_encoder(surface_data)

        surface_feat = local_feat
        for conv in self.patch_encoder:
            surface_feat = conv(
                surface_feat,
                surface_data['vertices'],
                surface_data['normals'],
                surface_data['neighbors']
            )

        residue_indices = surface_data['residue_indices']

        max_residues = []
        for b in range(B):
            valid_indices = residue_indices[b][residue_indices[b] > 0]
            max_residues.append(valid_indices.max().item())
        num_residues = max(max_residues)

        residue_queries = torch.zeros(B, num_residues, self.hidden_dim).to(surface_feat.device)

        residue_features = []
        for b in range(B):
            mask = torch.zeros(N).to(surface_feat.device)
            for i in range(1, num_residues + 1):
                residue_points = (residue_indices[b] == i)
                if residue_points.any():
                    mask[residue_points] = 1

            res_feat, _ = self.residue_pooling(
                query=residue_queries[b].unsqueeze(0),
                key=surface_feat[b].unsqueeze(0),
                value=surface_feat[b].unsqueeze(0),
                key_padding_mask=~mask.bool().unsqueeze(0)
            )
            residue_features.append(res_feat.squeeze(0))

        residue_features = torch.stack(residue_features, dim=0)

        enhanced_surface_feat = self.cross_residue_conv(
            surface_feat,
            surface_data['relative_positions'],
            surface_data['local_frames'],
            surface_data['neighbors']
        )

        residue_features = self.residue_norm(residue_features)
        enhanced_surface_feat = self.surface_norm(enhanced_surface_feat)

        return residue_features, enhanced_surface_feat

class ProteinEncoder(nn.Module):

    def __init__(self, hidden_dim: int, saprot_model=None, num_structure_layers: int = 3, structure_drop_rate: float = 0.1):

        super().__init__()
        self.hidden_dim = hidden_dim

        if saprot_model is None:
            saprot_path = os.path.abspath("./models/saprot")
            logger.info(f"Loading SaProt model from {saprot_path}")

            self.saprot = AutoModel.from_pretrained(saprot_path)
            self.tokenizer = AutoTokenizer.from_pretrained(saprot_path)
            logger.info("Successfully loaded SaProt model")
        else:
            self.saprot = saprot_model
            self.tokenizer = AutoTokenizer.from_pretrained("./models/saprot")
            logger.info("Using provided SaProt model")

        self.seq_proj = nn.Linear(self.saprot.config.hidden_size, hidden_dim)

        self.structure_encoder = StructureEncoderGVP(
            hidden_dim=hidden_dim,
            num_layers=num_structure_layers,
            drop_rate=structure_drop_rate
        )

        self.surface_encoder = SurfaceGNNEncoder(hidden_dim=hidden_dim)

        self.fusion = MultiModalFusion(hidden_dim=hidden_dim, use_structure=True, use_surface=True)

    def forward(self, features: Dict) -> torch.Tensor:

        combined_seqs = features['sequence']['combined_seq']
        device = features['structure'].device
        inputs = self.tokenizer(
            combined_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if torch.is_grad_enabled() and any(param.requires_grad for param in self.saprot.parameters()):
            outputs = self.saprot(**inputs)
        else:
            with torch.no_grad():
                outputs = self.saprot(**inputs)

        seq_features = outputs.last_hidden_state
        seq_features = seq_features[:, 1:-1, :]
        seq_features = self.seq_proj(seq_features)

        struct_features = self.structure_encoder(
            features['structure']
        )
        structure_lengths = features['structure'].batch_num_nodes().to(seq_features.device)
        seq_mask = torch.arange(
            seq_features.size(1),
            device=seq_features.device
        )[None, :] < structure_lengths[:, None]
        structure_mask = torch.arange(
            struct_features.size(1),
            device=seq_features.device
        )[None, :] < structure_lengths[:, None]

        surface_features = self.surface_encoder({
            'vertices': features['surface']['vertices'],
            'normals': features['surface']['normals'],
            'distance_features': features['surface']['distance_features'],
            'angle_features': features['surface']['angle_features'],
            'curvature': features['surface']['curvature'],
            'local_frames': features['surface']['local_frames'],
            'relative_positions': features['surface']['relative_positions'],
            'charges': features['surface']['charges'],
            'atom_types': features['surface']['atom_types'],
            'neighbors': features['surface']['neighbors'],
            'residue_indices': features['surface']['residue_indices'],
            'masks': features['surface']['masks'],
            'structure_feature': struct_features
        })

        fused_features = self.fusion({
            'sequence': {
                'combined_seq': seq_features,
                'mask': seq_mask
            },
            'structure': struct_features,
            'structure_mask': structure_mask,
            'surface': surface_features,
            'surface_mask': structure_mask[:, :surface_features.size(1)]
        })

        return fused_features
