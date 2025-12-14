"""notch pathway simulation.

primary entrypoints:
- notch.ode: continuous ode model (multicellular)
- notch.boolean: logical/boolean synchronous updates
- notch.cli: command-line interface
"""

from .boolean import simulate_boolean
from .graph import load_edge_list
from .ode import simulate_ode

__all__ = ["simulate_ode", "simulate_boolean", "load_edge_list"]
