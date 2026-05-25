

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoModel

from .encoders import ProteinEncoder

logger = logging.getLogger(__name__)

class DBP2Predictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 256,
        num_neighbors: int = 8,
        dropout: float = 0.2
    ):
        
        super().__init__()
        
        self.saprot = AutoModel.from_pretrained('./models/saprot')
        logger.info("Successfully loaded SaProt model")
        
        self.protein_encoder = ProteinEncoder(
            hidden_dim=hidden_dim,
            saprot_model=self.saprot
        )
        
        self.contact_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        self.nucleotide_preference = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        self.binding_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        fused_feat = self.protein_encoder(features)
        
        batch_size = fused_feat.size(0)
        num_residues = fused_feat.size(1)
        
        contact_cls = self.contact_classifier(fused_feat)
        contact_cls = contact_cls.view(batch_size, num_residues, 4)
        
        nuc_preference = self.nucleotide_preference(fused_feat)
        nuc_preference = nuc_preference.view(batch_size, num_residues, 4)
        
        binding_importance = self.binding_importance(fused_feat)
        binding_importance = binding_importance.view(batch_size, num_residues)
        
        return {
            'contact_cls': contact_cls,
            'nuc_preference': nuc_preference,
            'binding_importance': binding_importance,
            'residue_feat': fused_feat
        }