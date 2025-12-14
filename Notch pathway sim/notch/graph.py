from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _parse_ints(line: str) -> tuple[int, ...]:
    # allow: "i j", "i,j", "i\tj".
    parts = [p for p in line.replace(",", " ").split() if p]
    return tuple(int(p) for p in parts)


def load_edge_list(path: str | Path, *, n: int | None = None, undirected: bool = True) -> list[list[int]]:
    """load adjacency list from an edge list.

    file format:
    - one edge per line: "i j" (0-based indices)
    - blank lines and lines starting with '#' are ignored

    if n is not provided, it is inferred as max_index+1.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    edges: list[tuple[int, int]] = []
    max_idx = -1

    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ints = _parse_ints(line)
        if len(ints) < 2:
            raise ValueError(f"invalid edge line: {raw!r}")
        i, j = ints[0], ints[1]
        if i < 0 or j < 0:
            raise ValueError(f"negative node index in line: {raw!r}")
        edges.append((i, j))
        max_idx = max(max_idx, i, j)

    if n is None:
        n = max_idx + 1

    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        if i >= n or j >= n:
            raise ValueError(f"edge ({i},{j}) out of bounds for n={n}")
        if j != i:
            adj[i].append(j)
        if undirected and i != j:
            adj[j].append(i)

    # de-dup while preserving order
    for k in range(n):
        seen: set[int] = set()
        out: list[int] = []
        for v in adj[k]:
            if v not in seen:
                out.append(v)
                seen.add(v)
        adj[k] = out

    return adj


@dataclass(frozen=True)
class Adjacency:
    """explicit adjacency wrapper to avoid ambiguity."""

    neighbors: list[list[int]]

    @property
    def n(self) -> int:
        return len(self.neighbors)
