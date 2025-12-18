"""
Axon Guidance Simulation with Eph/Ephrin Gradients

Models growth cone navigation through ephrin gradients, including
attraction, repulsion, and topographic map formation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GrowthCone:
    """Represents a growth cone with position and direction"""
    position: np.ndarray  # [x, y] coordinates
    direction: float  # Angle in radians
    history: List[np.ndarray]  # Trajectory history
    active: bool = True  # Whether still growing
    
    def __post_init__(self):
        if not self.history:
            self.history = [self.position.copy()]


@dataclass
class AxonGuidanceParameters:
    """Parameters for axon guidance model"""
    # Growth cone dynamics
    growth_speed: float = 1.0  # Base extension speed
    turning_sensitivity: float = 0.5  # Response to gradients
    persistence_length: float = 10.0  # How straight growth cones grow
    
    # Gradient sensing
    filopodia_length: float = 3.0  # Reach of sensing
    sensing_noise: float = 0.1  # Stochasticity in sensing
    
    # Response types
    attractive_gain: float = 1.0  # Attraction to low ephrin
    repulsive_gain: float = 2.0  # Repulsion from high ephrin
    
    # Stopping conditions
    max_steps: int = 200  # Maximum growth steps
    target_distance: float = 5.0  # Distance to stop at target
    
    # Collision avoidance
    self_avoidance: bool = True
    avoidance_radius: float = 2.0


class EphrinGradient:
    """
    Generates various ephrin gradient patterns for axon guidance
    """
    
    @staticmethod
    def linear_gradient(grid_size: Tuple[int, int], 
                       direction: str = 'horizontal',
                       min_val: float = 0.0, 
                       max_val: float = 10.0) -> np.ndarray:
        """Create linear ephrin gradient"""
        h, w = grid_size
        
        if direction == 'horizontal':
            gradient = np.linspace(min_val, max_val, w)
            field = np.tile(gradient, (h, 1))
        elif direction == 'vertical':
            gradient = np.linspace(min_val, max_val, h)
            field = np.tile(gradient.reshape(-1, 1), (1, w))
        else:
            raise ValueError("direction must be 'horizontal' or 'vertical'")
        
        return field
    
    @staticmethod
    def exponential_gradient(grid_size: Tuple[int, int],
                            source_position: Tuple[int, int],
                            decay_length: float = 20.0,
                            amplitude: float = 10.0) -> np.ndarray:
        """Create exponential decay gradient from a source"""
        h, w = grid_size
        y, x = np.ogrid[:h, :w]
        sy, sx = source_position
        
        distance = np.sqrt((x - sx)**2 + (y - sy)**2)
        field = amplitude * np.exp(-distance / decay_length)
        
        return field
    
    @staticmethod
    def opposing_gradients(grid_size: Tuple[int, int],
                          ephrin_a_max: float = 10.0,
                          ephrin_b_max: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create opposing gradients (e.g., EphrinA high->low, EphrinB low->high)
        Used in topographic mapping
        """
        h, w = grid_size
        
        # EphrinA: high on left, low on right
        ephrin_a = np.linspace(ephrin_a_max, 0, w)
        ephrin_a_field = np.tile(ephrin_a, (h, 1))
        
        # EphrinB: low on left, high on right
        ephrin_b = np.linspace(0, ephrin_b_max, w)
        ephrin_b_field = np.tile(ephrin_b, (h, 1))
        
        return ephrin_a_field, ephrin_b_field
    
    @staticmethod
    def striped_pattern(grid_size: Tuple[int, int],
                       stripe_width: int = 10,
                       high_val: float = 10.0,
                       low_val: float = 0.0) -> np.ndarray:
        """Create alternating stripes of high/low ephrin"""
        h, w = grid_size
        field = np.zeros((h, w))
        
        for i in range(0, w, stripe_width * 2):
            field[:, i:i+stripe_width] = high_val
        
        return field
    
    @staticmethod
    def circular_barrier(grid_size: Tuple[int, int],
                        center: Tuple[int, int],
                        inner_radius: float = 20.0,
                        outer_radius: float = 30.0,
                        barrier_height: float = 10.0) -> np.ndarray:
        """Create circular ephrin barrier"""
        h, w = grid_size
        y, x = np.ogrid[:h, :w]
        cy, cx = center
        
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        field = np.zeros((h, w))
        
        mask = (distance >= inner_radius) & (distance <= outer_radius)
        field[mask] = barrier_height
        
        return field


class AxonGuidanceSimulation:
    """
    Simulates axon growth cone navigation through ephrin gradients
    """
    
    def __init__(self, ephrin_field: np.ndarray, 
                 params: AxonGuidanceParameters = None):
        self.ephrin_field = ephrin_field
        self.params = params or AxonGuidanceParameters()
        self.grid_size = ephrin_field.shape
        self.growth_cones: List[GrowthCone] = []
        
    def add_growth_cone(self, position: Tuple[float, float], 
                       direction: float = 0.0):
        """Add a growth cone to the simulation"""
        gc = GrowthCone(
            position=np.array(position, dtype=float),
            direction=direction,
            history=[]
        )
        self.growth_cones.append(gc)
    
    def sense_gradient(self, position: np.ndarray, direction: float) -> float:
        """
        Sense ephrin gradient using filopodia
        
        Returns:
            Gradient direction (positive = turn right, negative = turn left)
        """
        p = self.params
        
        # Sample ephrin at current position
        current_ephrin = self._sample_field(position)
        
        # Sample in front
        front_angle = direction
        front_pos = position + p.filopodia_length * np.array([
            np.cos(front_angle), np.sin(front_angle)
        ])
        front_ephrin = self._sample_field(front_pos)
        
        # Sample left and right
        left_angle = direction + np.pi / 4
        left_pos = position + p.filopodia_length * np.array([
            np.cos(left_angle), np.sin(left_angle)
        ])
        left_ephrin = self._sample_field(left_pos)
        
        right_angle = direction - np.pi / 4
        right_pos = position + p.filopodia_length * np.array([
            np.cos(right_angle), np.sin(right_angle)
        ])
        right_ephrin = self._sample_field(right_pos)
        
        # Calculate turning bias
        # Positive gradient = repulsion (turn away from high ephrin)
        gradient_left_right = right_ephrin - left_ephrin
        
        # Add noise
        gradient_left_right += np.random.normal(0, p.sensing_noise)
        
        return gradient_left_right
    
    def _sample_field(self, position: np.ndarray) -> float:
        """Sample ephrin field at a position with bilinear interpolation"""
        x, y = position
        h, w = self.grid_size
        
        # Boundary check
        if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            return 0.0
        
        # Bilinear interpolation
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        
        dx, dy = x - x0, y - y0
        
        value = (self.ephrin_field[y0, x0] * (1 - dx) * (1 - dy) +
                self.ephrin_field[y0, x1] * dx * (1 - dy) +
                self.ephrin_field[y1, x0] * (1 - dx) * dy +
                self.ephrin_field[y1, x1] * dx * dy)
        
        return value
    
    def update_growth_cone(self, gc: GrowthCone) -> bool:
        """
        Update growth cone position and direction
        
        Returns:
            Whether growth cone is still active
        """
        if not gc.active:
            return False
        
        p = self.params
        
        # Sense gradient
        gradient_signal = self.sense_gradient(gc.position, gc.direction)
        
        # Update direction based on repulsive response
        turning_rate = p.turning_sensitivity * p.repulsive_gain * gradient_signal
        
        # Add persistence (tendency to go straight)
        persistence_noise = np.random.normal(0, 1.0 / p.persistence_length)
        
        gc.direction += turning_rate + persistence_noise
        
        # Update position
        displacement = p.growth_speed * np.array([
            np.cos(gc.direction),
            np.sin(gc.direction)
        ])
        
        new_position = gc.position + displacement
        
        # Check boundaries
        h, w = self.grid_size
        if (new_position[0] < 0 or new_position[0] >= w or
            new_position[1] < 0 or new_position[1] >= h):
            gc.active = False
            return False
        
        # Check collision with own trajectory (self-avoidance)
        if p.self_avoidance:
            for past_pos in gc.history[-20:]:  # Check recent history
                if np.linalg.norm(new_position - past_pos) < p.avoidance_radius:
                    # Try to turn away
                    gc.direction += np.random.choice([-1, 1]) * np.pi / 4
                    return True
        
        gc.position = new_position
        gc.history.append(new_position.copy())
        
        # Check if maximum steps reached
        if len(gc.history) >= p.max_steps:
            gc.active = False
            return False
        
        return True
    
    def simulate_step(self) -> int:
        """
        Simulate one time step for all growth cones
        
        Returns:
            Number of active growth cones
        """
        active_count = 0
        
        for gc in self.growth_cones:
            if self.update_growth_cone(gc):
                active_count += 1
        
        return active_count
    
    def simulate(self, max_iterations: int = 500) -> List[List[np.ndarray]]:
        """
        Run full simulation until all growth cones stop
        
        Returns:
            Trajectories for all growth cones
        """
        for iteration in range(max_iterations):
            active_count = self.simulate_step()
            
            if active_count == 0:
                break
        
        # Return all trajectories
        return [gc.history for gc in self.growth_cones]
    
    def get_trajectory_array(self, gc_index: int = 0) -> np.ndarray:
        """Get trajectory as numpy array for plotting"""
        if gc_index >= len(self.growth_cones):
            return np.array([])
        
        history = self.growth_cones[gc_index].history
        return np.array(history)


class TopographicMapping:
    """
    Models topographic map formation through Eph/Ephrin gradients
    (e.g., retinotectal mapping)
    """
    
    def __init__(self, source_size: int = 50, target_size: int = 100):
        self.source_size = source_size  # Retina
        self.target_size = target_size  # Tectum
        
        # Create opposing gradients in target
        self.target_ephrin_a, self.target_ephrin_b = \
            EphrinGradient.opposing_gradients(
                (target_size, target_size),
                ephrin_a_max=10.0,
                ephrin_b_max=10.0
            )
    
    def generate_axon_source_positions(self, n_axons: int) -> List[Tuple[int, int]]:
        """Generate axon starting positions across the source (retina)"""
        positions = []
        
        for i in range(n_axons):
            # Distribute evenly across source
            row = int((i * self.source_size) / n_axons)
            col = self.source_size // 2  # Start from middle
            positions.append((col, row))
        
        return positions
    
    def calculate_target_position(self, source_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate where an axon from source should terminate in target
        based on matching gradients
        """
        # Normalize source position (0 to 1)
        source_y = source_pos[1] / self.source_size
        
        # Map to target position
        # Axons from anterior (low y) -> posterior (high y) tectum
        target_y = int(source_y * self.target_size)
        target_x = self.target_size // 2
        
        return (target_x, target_y)
    
    def simulate_topographic_map(self, n_axons: int = 20,
                                params: AxonGuidanceParameters = None) -> List[List[np.ndarray]]:
        """
        Simulate formation of topographic map with multiple axons
        """
        source_positions = self.generate_axon_source_positions(n_axons)
        
        # Use EphrinA gradient for anterior-posterior guidance
        sim = AxonGuidanceSimulation(self.target_ephrin_a, params)
        
        # Add growth cones starting from left edge of target
        for source_y in range(0, self.source_size, self.source_size // n_axons):
            start_x = 5  # Left edge
            start_y = int((source_y / self.source_size) * self.target_size)
            
            # Initial direction: toward right
            sim.add_growth_cone((start_x, start_y), direction=0.0)
        
        # Run simulation
        trajectories = sim.simulate(max_iterations=500)
        
        return trajectories
