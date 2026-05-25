from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1, chunk_size: int = 128):
        
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        # Cache the latest attention tensors for interpretation.
        self._last_attentions = None  # List[Tensor] with shape [B*H, chunk, Lk]
        self._last_attention_cat = None  # Tensor with shape [B, H, Lq, Lk]
        self._last_scores_cat = None  # Tensor with shape [B, H, Lq, Lk]
        self._last_shapes = None  # (batch_size, seq_len_q, seq_len_k)
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand(batch_size, self.num_heads, seq_len_q, seq_len_k)
            mask = mask.reshape(batch_size * self.num_heads, seq_len_q, seq_len_k)
        
        q = q.reshape(batch_size * self.num_heads, seq_len_q, self.head_dim)
        k = k.reshape(batch_size * self.num_heads, seq_len_k, self.head_dim)
        v = v.reshape(batch_size * self.num_heads, seq_len_k, self.head_dim)
        
        outputs = []
        attentions = []
        scores_list = []
        
        for i in range(0, seq_len_q, self.chunk_size):
            q_chunk = q[:, i:i+self.chunk_size]
            
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                mask_chunk = mask[:, i:i+self.chunk_size]
                scores = scores.masked_fill(mask_chunk, float('-inf'))
            
            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            
            chunk_output = torch.matmul(attention, v)
            outputs.append(chunk_output)
            if attention.requires_grad:
                attention.retain_grad()
            attentions.append(attention)
            scores_list.append(scores)
        
        x = torch.cat(outputs, dim=1)
        attention = torch.cat(attentions, dim=1)
        
        x = x.view(batch_size, self.num_heads, seq_len_q, self.head_dim)
        attention = attention.view(batch_size, self.num_heads, seq_len_q, seq_len_k)
        scores_cat = torch.cat(scores_list, dim=1)
        scores_cat = scores_cat.view(batch_size, self.num_heads, seq_len_q, seq_len_k)
        if attention.requires_grad:
            attention.retain_grad()
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len_q, self.hidden_dim)
        
        output = self.o_proj(x)
        self._last_attentions = attentions
        self._last_attention_cat = attention
        self._last_scores_cat = scores_cat
        self._last_shapes = (batch_size, seq_len_q, seq_len_k)
        
        return output, attention

class CrossModalAttention(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_mask: Optional[torch.Tensor] = None,
        x2_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        self_mask = None
        if x1_mask is not None:
            self_mask = ~x1_mask[:, None, :].expand(-1, x1.size(1), -1)

        cross_mask = None
        if x2_mask is not None:
            cross_mask = ~x2_mask[:, None, :].expand(-1, x1.size(1), -1)

        attended_x1, _ = self.attention(x1, x1, x1, self_mask)
        x1 = self.norm1(x1 + attended_x1)
        
        attended_x2, _ = self.attention(x1, x2, x2, cross_mask)
        x1 = self.norm2(x1 + attended_x2)

        if x1_mask is not None:
            x1 = x1 * x1_mask.unsqueeze(-1).to(dtype=x1.dtype)
        
        return x1

class MultiModalFusion(nn.Module):
    def __init__(self, hidden_dim: int, use_structure: bool = True, use_surface: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_structure = use_structure
        self.use_surface = use_surface
        
        if use_structure:
            self.seq_struct_fusion = CrossModalAttention(hidden_dim)
        if use_surface:
            self.seq_surface_fusion = CrossModalAttention(hidden_dim)
        
        input_dim = hidden_dim
        if use_structure:
            input_dim += hidden_dim
        if use_surface:
            input_dim += hidden_dim
        
        self.out_proj = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, features):
        seq_feat = features['sequence']['combined_seq']
        seq_mask = features['sequence'].get('mask')
        struct_feat = features['structure'] if 'structure' in features else None
        struct_mask = features.get('structure_mask')
        surface_feat = features['surface'] if 'surface' in features else None
        surface_mask = features.get('surface_mask')
        
        feats_to_concat = [seq_feat]
        
        if self.use_structure and struct_feat is not None:
            seq_struct_feat = self.seq_struct_fusion(seq_feat, struct_feat, seq_mask, struct_mask)
            feats_to_concat.append(seq_struct_feat)
        
        if self.use_surface and surface_feat is not None:
            seq_surface_feat = self.seq_surface_fusion(seq_feat, surface_feat, seq_mask, surface_mask)
            feats_to_concat.append(seq_surface_feat)
        
        fused_feat = torch.cat(feats_to_concat, dim=-1)
        
        return self.out_proj(fused_feat)
