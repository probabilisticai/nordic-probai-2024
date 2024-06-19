import copy
from dataclasses import dataclass
from typing import Any, List, Optional
import torch


@dataclass
class DataBatch:
    # Equclidean data to be diffused
    x: torch.LongTensor

    # Invariant and optional data to be diffused
    h: Optional[torch.LongTensor] = None

    # Indices for batching
    batch: Optional[torch.LongTensor] = None

    # Additional data used for conditioning
    context: Optional[torch.FloatTensor] = None

    # Edge indices between nodes
    edge_index: Optional[torch.LongTensor] = None

    def to(self, device: torch.device) -> "DataBatch":
        self.x = self.x.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.h is not None:
            self.h = self.h.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        return copy.deepcopy(self)

    def keys(self) -> List[str]:
        # return "x", and optionally "batch" and "h" if they are not None
        return [k for k in self.__dict__.keys() if self.__dict__[k] is not None]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
