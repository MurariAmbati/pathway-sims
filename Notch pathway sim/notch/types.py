from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray

GridTopology = Literal["von_neumann", "moore"]
Boundary = Literal["periodic", "reflect"]


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    topology: GridTopology = "von_neumann"
    boundary: Boundary = "periodic"

    @property
    def n(self) -> int:
        return self.rows * self.cols
