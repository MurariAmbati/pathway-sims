"""
simulation engine - integrates all components
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from models import (
    ChemokineGradient,
    LeukocytePopulation,
    TissueCompartment
)


@dataclass
class SimulationConfig:
    """simulation parameters"""
    duration: float = 600.0  # seconds
    dt: float = 0.1  # time step (s)
    save_interval: float = 1.0  # data save interval (s)
    
    # numerical stability
    max_dt: float = 1.0
    adaptive_timestep: bool = False


@dataclass
class SimulationState:
    """current simulation state"""
    time: float = 0.0
    step: int = 0
    
    # data storage
    cell_positions: List[np.ndarray] = field(default_factory=list)
    cell_states: List[List[str]] = field(default_factory=list)
    gradient_snapshots: List[np.ndarray] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    
    # statistics
    mean_speeds: List[float] = field(default_factory=list)
    chemotactic_indices: List[float] = field(default_factory=list)


class SimulationEngine:
    """
    main simulation engine
    coordinates gradient evolution, cell migration, and data collection
    """
    
    def __init__(self,
                 tissue: TissueCompartment,
                 gradients: Dict[str, ChemokineGradient],
                 population: LeukocytePopulation,
                 config: SimulationConfig = None):
        """
        args:
            tissue: tissue microenvironment
            gradients: dict of chemokine gradients
            population: leukocyte population
            config: simulation configuration
        """
        self.tissue = tissue
        self.gradients = gradients
        self.population = population
        self.config = config or SimulationConfig()
        
        self.state = SimulationState()
        
        # performance tracking
        self.last_save_time = 0.0
        self.computation_time = 0.0
    
    def step(self):
        """
        single simulation time step
        """
        dt = self.config.dt
        
        # update chemokine gradients
        for gradient in self.gradients.values():
            gradient.step(dt)
        
        # update cell population
        boundaries = self.tissue.get_boundaries()
        self.population.step(self.gradients, dt, boundaries)
        
        # increment time
        self.state.time += dt
        self.state.step += 1
    
    def save_state(self):
        """
        save current state to history
        """
        # cell positions
        positions = self.population.get_positions()
        self.state.cell_positions.append(positions.copy())
        
        # cell states
        states = self.population.get_states()
        self.state.cell_states.append(states)
        
        # time
        self.state.time_points.append(self.state.time)
        
        # statistics
        mean_speed = self.population.get_mean_speed()
        self.state.mean_speeds.append(mean_speed)
        
        # gradient snapshot (downsample for memory)
        if len(self.gradients) > 0:
            gradient = list(self.gradients.values())[0]
            snapshot = gradient.concentration[::4, ::4, ::4].copy()
            self.state.gradient_snapshots.append(snapshot)
        
        self.last_save_time = self.state.time
    
    def run(self, progress_callback=None):
        """
        run complete simulation
        
        args:
            progress_callback: function(time, progress) called each save
        """
        self.computation_time = time.time()
        
        # initial state
        self.save_state()
        
        # main loop
        n_steps = int(self.config.duration / self.config.dt)
        
        for _ in range(n_steps):
            self.step()
            
            # save at intervals
            if self.state.time - self.last_save_time >= self.config.save_interval:
                self.save_state()
                
                # progress callback
                if progress_callback is not None:
                    progress = self.state.time / self.config.duration
                    progress_callback(self.state.time, progress)
        
        # final save
        self.save_state()
        
        self.computation_time = time.time() - self.computation_time
    
    def get_results(self) -> Dict:
        """
        return simulation results
        """
        return {
            'time': np.array(self.state.time_points),
            'positions': self.state.cell_positions,
            'states': self.state.cell_states,
            'mean_speeds': np.array(self.state.mean_speeds),
            'chemotactic_indices': np.array(self.state.chemotactic_indices),
            'gradients': self.state.gradient_snapshots,
            'duration': self.state.time,
            'n_cells': len(self.population.cells),
            'computation_time': self.computation_time
        }
    
    def get_summary_statistics(self) -> Dict:
        """
        compute summary statistics
        """
        final_positions = self.state.cell_positions[-1]
        
        # displacement statistics
        initial_positions = self.state.cell_positions[0]
        displacements = np.linalg.norm(final_positions - initial_positions, axis=1)
        
        # state distribution
        state_counts = {}
        for state_list in self.state.cell_states:
            for state in state_list:
                state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            'mean_displacement': np.mean(displacements),
            'std_displacement': np.std(displacements),
            'max_displacement': np.max(displacements),
            'mean_speed': np.mean(self.state.mean_speeds),
            'state_distribution': state_counts,
            'total_cells': len(self.population.cells),
            'simulation_duration': self.state.time,
            'timesteps': self.state.step
        }


def create_neutrophil_recruitment_simulation(
    n_neutrophils: int = 50,
    duration: float = 300.0
) -> SimulationEngine:
    """
    setup neutrophil recruitment to inflammation site
    
    args:
        n_neutrophils: number of cells
        duration: simulation duration (s)
    
    returns:
        configured SimulationEngine
    """
    from models import (
        create_inflammation_tissue,
        create_inflammation_gradient,
        create_neutrophil_receptors,
        NEUTROPHIL_PROPS
    )
    
    # tissue
    tissue = create_inflammation_tissue(size=(200, 200, 100), grid_spacing=2.0)
    
    # gradient
    gradient = create_inflammation_gradient(tissue.geometry.grid_size, 
                                          tissue.geometry.grid_spacing)
    
    # equilibrate gradient
    for _ in range(1000):
        gradient.step(0.1)
    
    gradients = {"CXCR1": gradient, "CXCR2": gradient}
    
    # population
    population = LeukocytePopulation()
    population.spawn_cells(
        n_cells=n_neutrophils,
        properties=NEUTROPHIL_PROPS,
        receptors_factory=create_neutrophil_receptors,
        spawn_region=((50, 150), (50, 150), (80, 95))
    )
    
    # configuration
    config = SimulationConfig(duration=duration, dt=0.1, save_interval=2.0)
    
    return SimulationEngine(tissue, gradients, population, config)


def create_t_cell_homing_simulation(
    n_tcells: int = 30,
    duration: float = 400.0
) -> SimulationEngine:
    """
    setup t-cell homing to lymph node
    """
    from models import (
        create_lymph_node_tissue,
        create_lymph_node_gradient,
        create_t_cell_receptors,
        T_CELL_PROPS
    )
    
    tissue = create_lymph_node_tissue(size=(300, 300, 200), grid_spacing=3.0)
    
    gradient = create_lymph_node_gradient(tissue.geometry.grid_size,
                                         tissue.geometry.grid_spacing)
    
    # equilibrate
    for _ in range(1000):
        gradient.step(0.1)
    
    gradients = {"CCR7": gradient}
    
    population = LeukocytePopulation()
    population.spawn_cells(
        n_cells=n_tcells,
        properties=T_CELL_PROPS,
        receptors_factory=create_t_cell_receptors,
        spawn_region=((100, 200), (100, 200), (150, 190))
    )
    
    config = SimulationConfig(duration=duration, dt=0.1, save_interval=2.0)
    
    return SimulationEngine(tissue, gradients, population, config)
