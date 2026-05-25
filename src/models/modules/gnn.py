

import functools
from typing import Dict, List, Optional, Tuple

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


def tuple_sum(*args):
    
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    valid_v_args = [v for v in v_args if v is not None and v.numel() > 0]
    if not valid_v_args:
         return torch.cat(s_args, dim=dim), None
    vec_dim = dim - 1 if dim > 0 else dim
    device = valid_v_args[0].device
    processed_v_args = []
    for v in v_args:
        if v is None or v.numel() == 0:
             pass
        else:
            processed_v_args.append(v.to(device))

    if not processed_v_args:
         return torch.cat(s_args, dim=dim), None

    return torch.cat(s_args, dim=dim), torch.cat(processed_v_args, dim=vec_dim)

def tuple_index(x, idx):
    
    if x[1] is None:
        return x[0][idx], None
    return x[0][idx], x[1][idx]

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    
    if x is None:
        return None
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    
    if nv == 0:
        return x, None
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    
    if v is None or v.shape[-2] == 0:
        return s
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP(nn.Module):
    
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super().__init__()
        self.input_dim_s, self.input_dim_v = in_dims
        self.output_dim_s, self.output_dim_v = out_dims
        self.vector_gate = vector_gate

        if self.input_dim_v > 0:
            self.h_dim = h_dim or max(self.input_dim_v, self.output_dim_v)
            self.wh = nn.Linear(self.input_dim_v, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.input_dim_s, self.output_dim_s)
            if self.output_dim_v > 0:
                self.wv = nn.Linear(self.h_dim, self.output_dim_v, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.output_dim_s, self.output_dim_v)
            else: 
                 self.wv = None
                 if self.vector_gate:
                      self.wsv = None

        else:
            self.ws = nn.Linear(self.input_dim_s, self.output_dim_s)
            if self.output_dim_v > 0:
                 self.wv = None
                 if self.vector_gate:
                      self.wsv = None
            else:
                 self.wv = None
                 if self.vector_gate:
                      self.wsv = None

        self.scalar_act, self.vector_act = activations

    def forward(self, x):
        if self.input_dim_v > 0:
            s, v = x
            if v is None:
                v = torch.zeros(s.shape[0], self.input_dim_v, 3, device=s.device, dtype=s.dtype)

            v_permuted = v.permute(*range(v.dim()-2), -1, -2)

            vh_permuted = self.wh(v_permuted)

            vh = vh_permuted.permute(*range(vh_permuted.dim()-2), -1, -2)

            vn = _norm_no_nan(vh, axis=-1)
            s_cat = torch.cat([s, vn], -1)
            s_out = self.ws(s_cat)

            if self.output_dim_v > 0 and self.wv is not None:
                 vh_permuted = vh.permute(*range(vh.dim()-2), -1, -2)

                 v_out_permuted = self.wv(vh_permuted)

                 v_out = v_out_permuted.permute(*range(v_out_permuted.dim()-2), -1, -2)

                 if self.vector_gate and self.wsv is not None:
                     if self.vector_act:
                         gate = self.wsv(self.vector_act(s_out))
                     else:
                         gate = self.wsv(s_out)
                     v_out = v_out * torch.sigmoid(gate).unsqueeze(-1)
                 elif self.vector_act:
                     v_out = v_out * self.vector_act(_norm_no_nan(v_out, axis=-1, keepdims=True))
            else:
                v_out = None

        else:
            s_out = self.ws(x)
            v_out = None

        if self.scalar_act:
            s_out = self.scalar_act(s_out)

        if self.output_dim_v == 0:
             v_out = None

        return (s_out, v_out) if v_out is not None else s_out

class _VDropout(nn.Module):
    
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if x is None: return None
        device = self.dummy_param.device
        if not self.training or self.drop_rate == 0.:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate + 1e-8)
        return x

class Dropout(nn.Module):
    
    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class GVPLayerNorm(nn.Module):
    
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s) if self.s > 0 else None

    def forward(self, x):
        
        if not isinstance(x, tuple):
            return self.scalar_norm(x) if self.scalar_norm else x

        s, v = x
        if v is None or self.v == 0:
             return self.scalar_norm(s) if self.scalar_norm else s, None

        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        v_norm = v / (vn + 1e-8)

        s_norm = self.scalar_norm(s) if self.scalar_norm else s

        return s_norm, v_norm

class GVP_DGL_Layer(nn.Module):
    
    def __init__(self, node_dims, edge_dims, out_dims, activations=(F.relu, torch.sigmoid), vector_gate=True, n_message_layers=3, drop_rate=0.1):
        super().__init__()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.out_dims = out_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        msg_in_dims = (2 * node_dims[0] + edge_dims[0], 2 * node_dims[1] + edge_dims[1])
        message_func_layers = []
        if n_message_layers == 1:
            message_func_layers.append(GVP_(msg_in_dims, out_dims, activations=(None, None)))
        else:
             message_func_layers.append(GVP_(msg_in_dims, out_dims))
             for _ in range(n_message_layers - 2):
                  message_func_layers.append(GVP_(out_dims, out_dims))
             message_func_layers.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*message_func_layers)

    def edge_udf(self, edges):
        
        s_j, v_j = edges.src['h_s'], edges.src['h_v']
        s_i, v_i = edges.dst['h_s'], edges.dst['h_v']
        edge_s, edge_v = edges.data['e_s'], edges.data['e_v']

        device = s_j.device
        if v_j is None: v_j = torch.zeros(s_j.shape[0], self.node_dims[1], 3, device=device, dtype=s_j.dtype)
        if v_i is None: v_i = torch.zeros(s_i.shape[0], self.node_dims[1], 3, device=device, dtype=s_i.dtype)
        if edge_v is None: edge_v = torch.zeros(edge_s.shape[0], self.edge_dims[1], 3, device=device, dtype=edge_s.dtype)

        cat_s = torch.cat([s_j, edge_s, s_i], dim=-1)
        cat_v = torch.cat([v_j, edge_v, v_i], dim=-2)

        msg_s, msg_v = self.message_func((cat_s, cat_v))

        return {'msg_s': msg_s, 'msg_v': msg_v}

    def forward(self, g, node_feat, edge_feat):
        
        with g.local_scope():
            g.ndata['h_s'], g.ndata['h_v'] = node_feat
            g.edata['e_s'], g.edata['e_v'] = edge_feat

            g.apply_edges(self.edge_udf)

            g.update_all(message_func=fn.copy_e('msg_s', 'm_s'),
                         reduce_func=fn.mean('m_s', 'agg_s'))

            if self.out_dims[1] > 0:
                 g.update_all(message_func=fn.copy_e('msg_v', 'm_v'),
                              reduce_func=fn.mean('m_v', 'agg_v'))

            agg_s = g.ndata.pop('agg_s')
            agg_v = g.ndata.pop('agg_v') if self.out_dims[1] > 0 else None

            return agg_s, agg_v

class StructureEncoderGVP(nn.Module):

    def __init__(self, hidden_dim: int, num_layers: int = 3, drop_rate: float = 0.1):
        super().__init__()

        hs = hidden_dim // 2
        hv = hidden_dim // 6

        node_h_dim = (hs, hv)
        edge_h_dim = node_h_dim

        node_in_dim = (2, 3)
        edge_in_dim = (2, 1)

        self.W_v = GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True)
        self.W_e = GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True)

        self.gnn_layers = nn.ModuleList([
            GVP_DGL_Layer(
                node_dims=node_h_dim,
                edge_dims=edge_h_dim,
                out_dims=node_h_dim,
                drop_rate=drop_rate,
                vector_gate=True
            ) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            GVPLayerNorm(node_h_dim) for _ in range(num_layers)
        ])

        self.dropouts = nn.ModuleList([
            Dropout(drop_rate) for _ in range(num_layers)
        ])

        self.output_proj = GVP(node_h_dim, (hidden_dim, 0), activations=(F.relu, None), vector_gate=True)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        
        raw_node_feat = g.ndata['feat'].float()
        raw_edge_feat = g.edata['feat'].float()

        node_s = raw_node_feat[:, :2]
        node_v = raw_node_feat[:, 2:].reshape(-1, 3, 3)
        node_feat = (node_s, node_v)

        edge_s = torch.cat([raw_edge_feat[:, 0:1], raw_edge_feat[:, 4:5]], dim=-1)
        edge_v = raw_edge_feat[:, 1:4].unsqueeze(-2)
        edge_feat = (edge_s, edge_v)

        h = self.W_v(node_feat)
        e = self.W_e(edge_feat)

        for i in range(len(self.gnn_layers)):
            h_prev = h

            agg_msg_s, agg_msg_v = self.gnn_layers[i](g, h, e)
            agg_msg = (agg_msg_s, agg_msg_v)

            agg_msg = self.dropouts[i](agg_msg)

            h = tuple_sum(h_prev, agg_msg)

            h = self.norms[i](h)

        out = self.output_proj(h)

        batch_size = g.batch_size
        num_nodes_per_graph = g.batch_num_nodes().cpu().tolist()
        max_nodes = max(num_nodes_per_graph) if num_nodes_per_graph else 0

        padded_out = torch.zeros(batch_size, max_nodes, out.size(-1),
                               device=out.device, dtype=out.dtype)

        start_idx = 0
        for i, num_nodes in enumerate(num_nodes_per_graph):
            if num_nodes > 0:
                padded_out[i, :num_nodes] = out[start_idx : start_idx + num_nodes]
                start_idx += num_nodes

        return padded_out 