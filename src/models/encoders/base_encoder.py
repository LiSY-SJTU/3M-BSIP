from typing import Any, Dict

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):

    def __init__(self, hidden_dim: int):
        
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        
        raise NotImplementedError("Subclasses must implement forward")
    
    def validate_input(self, x: Dict[str, Any]) -> bool:
        
        return True 