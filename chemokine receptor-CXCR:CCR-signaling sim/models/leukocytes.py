"""
leukocyte agent-based model
implements cell migration, chemotaxis, and trafficking behaviors
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from models.receptors import ReceptorExpression
from models.signaling import SignalingNetwork, AdaptationModule
from models.chemokines import ChemokineGradient


class LeukocyteState(Enum):
    """leukocyte trafficking states"""
    CIRCULATING = "circulating"
    ROLLING = "rolling"
    ARRESTED = "arrested"
    MIGRATING = "migrating"
    ADHERED = "adhered"


@dataclass
class LeukocyteProperties:
    """biophysical properties of leukocytes"""
    cell_type: str
    radius: float  # μm
    basal_speed: float  # μm/min
    max_speed: float  # μm/min
    persistence_time: float  # s (directional memory)
    sensing_range: float  # μm (pseudopod extension)
    adhesion_threshold: float  # signal threshold for arrest


class Leukocyte:
    """
    agent-based leukocyte with chemotaxis and signaling
    implements persistent random walk with chemotactic bias
    """
    
    def __init__(self, 
                 properties: LeukocyteProperties,
                 receptors: ReceptorExpression,
                 position: np.ndarray):
        """
        args:
            properties: cell properties
            receptors: receptor expression profile
            position: initial position (x, y, z) in μm
        """
        self.props = properties
        self.receptors = receptors
        self.signaling = SignalingNetwork()
        self.adaptation = AdaptationModule()
        
        # spatial state
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.direction = self._random_direction()
        
        # trafficking state
        self.state = LeukocyteState.MIGRATING
        self.state_duration = 0.0
        
        # trajectory history
        self.trajectory = [self.position.copy()]
        self.time_history = [0.0]
        
        # internal state
        self.speed = properties.basal_speed / 60.0  # convert to μm/s
        self.polarization = 0.0
        self.adhesion_strength = 0.0
        
    def _random_direction(self) -> np.ndarray:
        """generate random unit vector"""
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        return np.array([x, y, z])
    
    def sense_gradients(self, gradients: Dict[str, ChemokineGradient]) -> Dict[str, float]:
        """
        sense chemokine concentrations at cell position
        
        args:
            gradients: dict mapping receptor name to gradient object
        
        returns:
            dict of concentrations for each receptor
        """
        concentrations = {}
        
        for receptor_name in self.receptors.receptors.keys():
            if receptor_name in gradients:
                gradient = gradients[receptor_name]
                conc_molecules = gradient.get_concentration(tuple(self.position))
                
                # convert to molar
                conc_molar = gradient.to_molar(conc_molecules)
                concentrations[receptor_name] = conc_molar
            else:
                concentrations[receptor_name] = 0.0
        
        return concentrations
    
    def compute_chemotactic_direction(self, 
                                      gradients: Dict[str, ChemokineGradient]) -> np.ndarray:
        """
        compute migration direction based on chemokine gradients
        uses spatial gradient sensing
        
        returns:
            unit direction vector
        """
        total_gradient = np.zeros(3)
        
        for receptor_name in self.receptors.receptors.keys():
            if receptor_name in gradients:
                gradient = gradients[receptor_name]
                grad_vec = gradient.get_gradient(tuple(self.position))
                
                # weight by receptor signal
                receptor = self.receptors.receptors[receptor_name]
                weight = receptor.signal_amplitude
                
                total_gradient += weight * grad_vec
        
        # normalize
        magnitude = np.linalg.norm(total_gradient)
        if magnitude > 1e-10:
            return total_gradient / magnitude
        else:
            return np.zeros(3)
    
    def update_direction(self, chemotactic_direction: np.ndarray, 
                        dt: float):
        """
        update migration direction with persistence and chemotactic bias
        implements biased persistent random walk
        
        args:
            chemotactic_direction: unit vector toward gradient
            dt: time step (s)
        """
        # random angular diffusion (rotational brownian motion)
        angular_diffusion = 1.0 / self.props.persistence_time
        noise_angle = np.sqrt(2 * angular_diffusion * dt)
        
        # add random perturbation
        noise = np.random.randn(3) * noise_angle
        noisy_direction = self.direction + noise
        
        # chemotactic bias (strength depends on signal)
        chemotactic_strength = self.signaling.get_migration_bias()
        biased_direction = (noisy_direction + 
                          chemotactic_strength * chemotactic_direction)
        
        # normalize
        magnitude = np.linalg.norm(biased_direction)
        if magnitude > 1e-10:
            self.direction = biased_direction / magnitude
        else:
            self.direction = self._random_direction()
        
        # update polarization state
        self.polarization = chemotactic_strength
    
    def update_position(self, dt: float, boundaries: Optional[Tuple] = None):
        """
        integrate position forward in time
        
        args:
            dt: time step (s)
            boundaries: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) or None
        """
        # speed modulation by signaling
        speed_factor = 1.0 + 0.5 * self.polarization
        current_speed = min(self.speed * speed_factor, 
                          self.props.max_speed / 60.0)
        
        # update position
        displacement = self.direction * current_speed * dt
        self.position += displacement
        self.velocity = displacement / dt
        
        # apply boundary conditions (reflecting)
        if boundaries is not None:
            for i in range(3):
                if self.position[i] < boundaries[i][0]:
                    self.position[i] = boundaries[i][0]
                    self.direction[i] = abs(self.direction[i])
                elif self.position[i] > boundaries[i][1]:
                    self.position[i] = boundaries[i][1]
                    self.direction[i] = -abs(self.direction[i])
        
        # record trajectory
        self.trajectory.append(self.position.copy())
    
    def update_signaling(self, dt: float):
        """
        update intracellular signaling state
        """
        # total receptor signal
        total_signal = self.receptors.get_total_signal()
        
        # adapt signal (implements gradient sensing)
        adapted_signal = self.adaptation.step(total_signal, dt)
        
        # integrate signaling network
        self.signaling.step(max(0, adapted_signal), dt)
        
        # update adhesion
        self.adhesion_strength = self.signaling.get_adhesion_strength()
    
    def update_trafficking_state(self):
        """
        update trafficking state based on adhesion signal
        """
        if self.state == LeukocyteState.MIGRATING:
            if self.adhesion_strength > self.props.adhesion_threshold:
                self.state = LeukocyteState.ARRESTED
                self.speed = 0.0
        
        elif self.state == LeukocyteState.ARRESTED:
            if self.adhesion_strength < self.props.adhesion_threshold * 0.5:
                self.state = LeukocyteState.MIGRATING
                self.speed = self.props.basal_speed / 60.0
    
    def step(self, gradients: Dict[str, ChemokineGradient], 
             dt: float, boundaries: Optional[Tuple] = None):
        """
        complete integration step for leukocyte
        
        args:
            gradients: dict of chemokine gradients
            dt: time step (s)
            boundaries: spatial boundaries
        """
        # sense environment
        concentrations = self.sense_gradients(gradients)
        
        # update receptors
        self.receptors.step_all(concentrations, dt)
        
        # update signaling
        self.update_signaling(dt)
        
        # update trafficking state
        self.update_trafficking_state()
        
        # compute migration direction
        if self.state == LeukocyteState.MIGRATING:
            chemo_direction = self.compute_chemotactic_direction(gradients)
            self.update_direction(chemo_direction, dt)
            self.update_position(dt, boundaries)
        
        # track time
        self.state_duration += dt
    
    def get_summary(self) -> Dict:
        """return current state summary"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'state': self.state.value,
            'polarization': self.polarization,
            'adhesion': self.adhesion_strength,
            'receptor_occupancy': {
                name: receptor.occupancy() 
                for name, receptor in self.receptors.receptors.items()
            }
        }


class LeukocytePopulation:
    """
    manages multiple leukocytes in simulation
    """
    
    def __init__(self):
        self.cells: List[Leukocyte] = []
        self.time = 0.0
    
    def add_cell(self, leukocyte: Leukocyte):
        """add leukocyte to population"""
        self.cells.append(leukocyte)
    
    def spawn_cells(self, 
                   n_cells: int,
                   properties: LeukocyteProperties,
                   receptors_factory: callable,
                   spawn_region: Tuple[Tuple, Tuple, Tuple]):
        """
        spawn multiple cells in region
        
        args:
            n_cells: number of cells
            properties: cell properties
            receptors_factory: function returning ReceptorExpression
            spawn_region: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        """
        for _ in range(n_cells):
            # random position in spawn region
            x = np.random.uniform(spawn_region[0][0], spawn_region[0][1])
            y = np.random.uniform(spawn_region[1][0], spawn_region[1][1])
            z = np.random.uniform(spawn_region[2][0], spawn_region[2][1])
            
            position = np.array([x, y, z])
            receptors = receptors_factory()
            
            cell = Leukocyte(properties, receptors, position)
            self.add_cell(cell)
    
    def step(self, gradients: Dict[str, ChemokineGradient], 
             dt: float, boundaries: Optional[Tuple] = None):
        """
        update all cells
        """
        for cell in self.cells:
            cell.step(gradients, dt, boundaries)
        
        self.time += dt
    
    def get_positions(self) -> np.ndarray:
        """return array of all cell positions"""
        return np.array([cell.position for cell in self.cells])
    
    def get_states(self) -> List[str]:
        """return list of trafficking states"""
        return [cell.state.value for cell in self.cells]
    
    def count_by_state(self) -> Dict[str, int]:
        """count cells in each state"""
        states = self.get_states()
        return {
            state.value: states.count(state.value)
            for state in LeukocyteState
        }
    
    def get_mean_speed(self) -> float:
        """compute mean speed of migrating cells"""
        speeds = []
        for cell in self.cells:
            if cell.state == LeukocyteState.MIGRATING:
                speeds.append(np.linalg.norm(cell.velocity))
        
        return np.mean(speeds) if speeds else 0.0
    
    def get_chemotactic_index(self, target_position: np.ndarray) -> float:
        """
        compute chemotactic index (directionality toward target)
        
        CI = (d_start - d_end) / total_path_length
        
        CI = 1: perfect chemotaxis
        CI = 0: random walk
        CI < 0: away from target
        """
        indices = []
        
        for cell in self.cells:
            if len(cell.trajectory) < 2:
                continue
            
            # initial and final distances to target
            d_start = np.linalg.norm(cell.trajectory[0] - target_position)
            d_end = np.linalg.norm(cell.trajectory[-1] - target_position)
            
            # total path length
            path_length = 0
            for i in range(len(cell.trajectory) - 1):
                path_length += np.linalg.norm(
                    cell.trajectory[i+1] - cell.trajectory[i]
                )
            
            if path_length > 0:
                ci = (d_start - d_end) / path_length
                indices.append(ci)
        
        return np.mean(indices) if indices else 0.0


# predefined cell types

NEUTROPHIL_PROPS = LeukocyteProperties(
    cell_type="neutrophil",
    radius=7.5,  # μm
    basal_speed=10.0,  # μm/min
    max_speed=25.0,
    persistence_time=5.0,  # s
    sensing_range=10.0,
    adhesion_threshold=0.3
)

T_CELL_PROPS = LeukocyteProperties(
    cell_type="t_cell",
    radius=6.0,
    basal_speed=12.0,
    max_speed=20.0,
    persistence_time=8.0,
    sensing_range=8.0,
    adhesion_threshold=0.35
)

MONOCYTE_PROPS = LeukocyteProperties(
    cell_type="monocyte",
    radius=8.5,
    basal_speed=8.0,
    max_speed=18.0,
    persistence_time=6.0,
    sensing_range=12.0,
    adhesion_threshold=0.25
)
