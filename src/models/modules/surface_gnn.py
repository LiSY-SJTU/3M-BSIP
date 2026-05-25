import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SurfaceGNNEncoder(nn.Module):
    
    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.default_residue_feature = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.default_residue_feature, std=0.02)
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(46, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            GATv2Conv(
                in_feats=hidden_dim,
                out_feats=hidden_dim // num_heads,
                num_heads=num_heads,
                residual=True,
                feat_drop=0.1
            ) for _ in range(3)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(3)
        ])
        
        self.residue_aggregator = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, surface_data):
        
        batched_graph, features, residue_masks = self._build_batch_graph(surface_data)
        
        for i, gnn in enumerate(self.gnn_layers):
            features = gnn(batched_graph, features).view(-1, self.num_heads * (self.hidden_dim // self.num_heads))
            features = self.norms[i](features)
            features = F.elu(features)
        
        residue_features = []
        
        structure_feature = surface_data.get('structure_feature')
        
        for i, res_indices in enumerate(residue_masks):
            sample_feat = features[batched_graph.ndata['batch'] == i]
            if sample_feat.size(0) != res_indices.size(0):
                raise RuntimeError(
                    "Surface feature/residue index length mismatch after padding removal: "
                    f"features={sample_feat.size(0)}, residue_indices={res_indices.size(0)}"
                )
            
            if structure_feature is not None:
                max_res_id = structure_feature[i].shape[0]
            
            res_feats = []
            for res_id in range(1, max_res_id + 1):
                mask = (res_indices == res_id)
                if mask.sum() > 0:
                    res_feats.append(sample_feat[mask].mean(dim=0))
                else:
                    res_feats.append(self.default_residue_feature)
            
            residue_features.append(torch.stack(res_feats))
        
        residue_features = self._pad_and_process(residue_features)
        return residue_features

    def _build_batch_graph(self, data):
        graphs = []
        all_features = []
        residue_masks = []
        surface_masks = []
        
        for i in range(len(data['distance_features'])):
            device = next(self.parameters()).device
            
            distance_features = data['distance_features'][i].to(device)
            angle_features = data['angle_features'][i].to(device)
            curvature = data['curvature'][i].to(device)
            charges = data['charges'][i].to(device)
            atom_types = data['atom_types'][i].to(device)
            neighbors = data['neighbors'][i].to(device)
            residue_indices = data['residue_indices'][i].to(device)
            masks = data['masks'][i].to(device)
            relative_positions = data['relative_positions'][i].to(device)
            valid_idx = masks.nonzero(as_tuple=False).squeeze(-1)

            distance_features = distance_features[valid_idx]
            angle_features = angle_features[valid_idx]
            curvature = curvature[valid_idx]
            charges = charges[valid_idx]
            atom_types = atom_types[valid_idx]
            residue_indices = residue_indices[valid_idx]
            relative_positions = relative_positions[valid_idx]

            index_map = torch.full((masks.size(0),), -1, dtype=torch.long, device=device)
            index_map[valid_idx] = torch.arange(valid_idx.numel(), device=device)
            neighbors = index_map[neighbors[valid_idx].long()]
            edge_mask = neighbors >= 0

            features = torch.cat([
                atom_types,
                charges.unsqueeze(-1),
                curvature.unsqueeze(-1),
                distance_features,
                angle_features,
                relative_positions.reshape(relative_positions.size(0), -1),
            ], dim=-1)
            features = self.feature_encoder(features)
            
            g = dgl.DGLGraph().to(device)
            g.add_nodes(features.size(0))
            
            src = torch.arange(features.size(0), dtype=torch.long, device=device).unsqueeze(1).expand_as(neighbors)[edge_mask]
            dst = neighbors[edge_mask].long()
            
            if dst.numel() > 0:
                assert dst.max() < features.size(0), f"Invalid neighbor index {dst.max()} >= {features.size(0)}"
                assert dst.min() >= 0, f"Negative neighbor index {dst.min()}"
            
            g.add_edges(src, dst)
            
            graphs.append(g)
            all_features.append(features)
            residue_masks.append(residue_indices)
            surface_masks.append(torch.ones(features.size(0), dtype=torch.bool, device=device))
        
        batched_graph = dgl.batch(graphs)
        device = next(self.parameters()).device
        batched_graph = batched_graph.to(device)
        
        batch_indices = torch.cat([
            torch.full((g.num_nodes(),), i, device=device) 
            for i, g in enumerate(graphs)
        ])
        batched_graph.ndata['batch'] = batch_indices
        batched_graph.ndata['mask'] = torch.cat(surface_masks).to(device)
        batched_graph = dgl.add_self_loop(batched_graph)
        
        dst_nodes = batched_graph.edges()[1]
        num_nodes = batched_graph.num_nodes()
        assert torch.all(dst_nodes < num_nodes), "Invalid edge target node"
        
        return batched_graph, torch.cat(all_features).to(device), residue_masks

    def _pad_and_process(self, features_list):
        max_len = max(f.size(0) for f in features_list)
        padded = []
        masks = []
        
        for f in features_list:
            pad_size = max_len - f.size(0)
            padded_f = F.pad(f, (0,0,0,pad_size))
            padded.append(padded_f)
            mask = torch.cat([torch.ones(f.size(0)), torch.zeros(pad_size)]).bool()
            masks.append(mask)
        
        padded = torch.stack(padded)
        masks = torch.stack(masks).to(padded.device)
        
        output = self.residue_aggregator(
            padded,
            src_key_padding_mask=~masks
        )
        return output 
