"""
Spatial Eph/Ephrin Model for Tissue Boundary Formation and Cell Sorting

Implements 2D spatial simulations of cell populations with Eph/Ephrin interactions,
including cell-cell adhesion, repulsion, and boundary formation dynamics.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
import scipy.ndimage as ndimage


@dataclass
class CellPopulation:
    """Represents a population of cells with specific Eph/Ephrin expression"""
    name: str
    eph_level: float  # Eph receptor expression
    ephrin_level: float  # Ephrin ligand expression
    color: str  # For visualization
    adhesion_strength: float = 1.0  # Homotypic adhesion


@dataclass
class SpatialParameters:
    """Parameters for spatial cell sorting model"""
    grid_size: Tuple[int, int] = (100, 100)
    
    # Cell adhesion
    homotypic_adhesion: float = 1.0  # Same cell type
    heterotypic_adhesion: float = 0.3  # Different cell types
    
    # Eph/Ephrin interaction
    repulsion_strength: float = 2.0  # Repulsion from Eph-Ephrin interaction
    repulsion_range: float = 3.0  # Interaction range in grid units
    
    # Cell motility
    base_motility: float = 0.5  # Random cell movement
    directed_motility: float = 1.5  # Response to repulsion
    
    # Diffusion
    diffusion_rate: float = 0.1  # Smoothing of cell distributions
    
    # Boundary formation
    boundary_sharpening: float = 0.5  # Enhanced separation at boundaries


class TissueBoundaryModel:
    """
    2D spatial model of tissue boundary formation through Eph/Ephrin signaling
    """
    
    def __init__(self, params: SpatialParameters = None):
        self.params = params or SpatialParameters()
        self.grid = np.zeros(self.params.grid_size)
        self.eph_field = np.zeros(self.params.grid_size)
        self.ephrin_field = np.zeros(self.params.grid_size)
        self.populations = []
        
    def initialize_populations(self, populations: List[CellPopulation],
                              layout: str = 'side_by_side'):
        """
        Initialize spatial distribution of cell populations
        
        Args:
            populations: List of cell populations
            layout: 'side_by_side', 'mixed', or 'concentric'
        """
        self.populations = populations
        h, w = self.params.grid_size
        
        if layout == 'side_by_side':
            # Split grid vertically
            split_points = np.linspace(0, w, len(populations) + 1, dtype=int)
            
            for i, pop in enumerate(populations):
                x_start = split_points[i]
                x_end = split_points[i + 1]
                
                # Assign cell type
                self.grid[:, x_start:x_end] = i + 1
                
                # Set Eph/Ephrin fields
                self.eph_field[:, x_start:x_end] = pop.eph_level
                self.ephrin_field[:, x_start:x_end] = pop.ephrin_level
                
        elif layout == 'mixed':
            # Random initial mixing
            cell_types = np.random.choice(
                len(populations), 
                size=self.params.grid_size,
                p=[1/len(populations)] * len(populations)
            )
            self.grid = cell_types + 1
            
            for i, pop in enumerate(populations):
                mask = self.grid == (i + 1)
                self.eph_field[mask] = pop.eph_level
                self.ephrin_field[mask] = pop.ephrin_level
                
        elif layout == 'concentric':
            # Concentric circles
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            
            radii = np.linspace(0, min(h, w) // 2, len(populations) + 1)
            
            for i, pop in enumerate(populations):
                r_inner = radii[i]
                r_outer = radii[i + 1]
                
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                mask = (distance >= r_inner) & (distance < r_outer)
                
                self.grid[mask] = i + 1
                self.eph_field[mask] = pop.eph_level
                self.ephrin_field[mask] = pop.ephrin_level
                
        # Add some noise to make it more realistic
        self.eph_field += np.random.normal(0, 0.1 * self.eph_field.mean(), 
                                          self.params.grid_size)
        self.ephrin_field += np.random.normal(0, 0.1 * self.ephrin_field.mean(),
                                             self.params.grid_size)
        self.eph_field = np.maximum(self.eph_field, 0)
        self.ephrin_field = np.maximum(self.ephrin_field, 0)
    
    def calculate_repulsion_field(self) -> np.ndarray:
        """
        Calculate repulsion forces from Eph-Ephrin interactions
        
        Returns:
            2D array of repulsion magnitudes
        """
        # Smooth fields to represent local interactions
        kernel_size = int(self.params.repulsion_range)
        eph_smooth = ndimage.gaussian_filter(self.eph_field, sigma=kernel_size)
        ephrin_smooth = ndimage.gaussian_filter(self.ephrin_field, sigma=kernel_size)
        
        # Repulsion proportional to product of Eph and Ephrin
        repulsion = self.params.repulsion_strength * eph_smooth * ephrin_smooth
        
        return repulsion
    
    def calculate_adhesion_energy(self) -> float:
        """
        Calculate total adhesion energy of the system
        Lower energy = more stable boundaries
        """
        energy = 0.0
        h, w = self.params.grid_size
        
        for i in range(h - 1):
            for j in range(w - 1):
                cell_type = self.grid[i, j]
                
                # Check neighbors
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    neighbor_type = self.grid[ni, nj]
                    
                    if cell_type == neighbor_type:
                        # Homotypic interaction
                        energy -= self.params.homotypic_adhesion
                    else:
                        # Heterotypic interaction (less favorable)
                        energy -= self.params.heterotypic_adhesion
        
        return energy
    
    def update_step(self, dt: float = 0.1):
        """
        Perform one time step of the spatial model
        """
        repulsion = self.calculate_repulsion_field()
        
        # Calculate gradients for directed motion
        grad_y, grad_x = np.gradient(repulsion)
        
        # Update cell positions using Metropolis-like dynamics
        new_grid = self.grid.copy()
        new_eph = self.eph_field.copy()
        new_ephrin = self.ephrin_field.copy()
        
        h, w = self.params.grid_size
        n_swaps = int(0.1 * h * w)  # Try 10% of cells per step
        
        for _ in range(n_swaps):
            # Random cell position
            i = np.random.randint(1, h - 1)
            j = np.random.randint(1, w - 1)
            
            # Choose random neighbor
            di, dj = np.random.choice([-1, 0, 1], size=2)
            ni, nj = i + di, j + dj
            
            if di == 0 and dj == 0:
                continue
            if ni < 0 or ni >= h or nj < 0 or nj >= w:
                continue
            
            # Calculate energy before swap
            E_before = self._local_energy(i, j, self.grid)
            E_before += self._local_energy(ni, nj, self.grid)
            
            # Swap cells
            temp_grid = self.grid.copy()
            temp_grid[i, j], temp_grid[ni, nj] = temp_grid[ni, nj], temp_grid[i, j]
            
            # Calculate energy after swap
            E_after = self._local_energy(i, j, temp_grid)
            E_after += self._local_energy(ni, nj, temp_grid)
            
            # Add repulsion bias
            repulsion_bias = repulsion[i, j] + repulsion[ni, nj]
            
            # Accept or reject based on energy and repulsion
            dE = E_after - E_before + 0.1 * repulsion_bias
            
            if dE < 0 or np.random.random() < np.exp(-dE):
                new_grid[i, j] = self.grid[ni, nj]
                new_grid[ni, nj] = self.grid[i, j]
                
                new_eph[i, j] = self.eph_field[ni, nj]
                new_eph[ni, nj] = self.eph_field[i, j]
                
                new_ephrin[i, j] = self.ephrin_field[ni, nj]
                new_ephrin[ni, nj] = self.ephrin_field[i, j]
        
        # Apply diffusion for smoothing
        self.eph_field = ndimage.gaussian_filter(new_eph, 
                                                 sigma=self.params.diffusion_rate)
        self.ephrin_field = ndimage.gaussian_filter(new_ephrin,
                                                    sigma=self.params.diffusion_rate)
        self.grid = new_grid
    
    def _local_energy(self, i: int, j: int, grid: np.ndarray) -> float:
        """Calculate local adhesion energy for a cell"""
        energy = 0.0
        cell_type = grid[i, j]
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                neighbor_type = grid[ni, nj]
                
                if cell_type == neighbor_type:
                    energy -= self.params.homotypic_adhesion
                else:
                    energy -= self.params.heterotypic_adhesion
        
        return energy
    
    def measure_boundary_sharpness(self) -> float:
        """
        Measure how sharp the boundaries are between cell types
        
        Returns:
            Sharpness metric (higher = sharper boundaries)
        """
        # Calculate gradient magnitude
        grad_y, grad_x = np.gradient(self.grid.astype(float))
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # High gradients indicate sharp boundaries
        return gradient_magnitude.mean()
    
    def measure_segregation_index(self) -> float:
        """
        Measure degree of segregation between cell types
        
        Returns:
            Index from 0 (mixed) to 1 (completely segregated)
        """
        h, w = self.params.grid_size
        same_neighbors = 0
        total_neighbors = 0
        
        for i in range(h):
            for j in range(w):
                cell_type = self.grid[i, j]
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        total_neighbors += 1
                        if self.grid[ni, nj] == cell_type:
                            same_neighbors += 1
        
        if total_neighbors == 0:
            return 0.0
        
        return same_neighbors / total_neighbors
    
    def simulate(self, n_steps: int, record_interval: int = 10) -> List[np.ndarray]:
        """
        Run simulation for multiple time steps
        
        Returns:
            List of grid snapshots at recording intervals
        """
        snapshots = [self.grid.copy()]
        
        for step in range(n_steps):
            self.update_step()
            
            if (step + 1) % record_interval == 0:
                snapshots.append(self.grid.copy())
        
        return snapshots
    
    def get_boundary_cells(self) -> np.ndarray:
        """
        Identify cells at population boundaries
        
        Returns:
            Binary mask of boundary cells
        """
        grad_y, grad_x = np.gradient(self.grid.astype(float))
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # Cells with high gradient are at boundaries
        boundary_mask = gradient_magnitude > 0.5
        
        return boundary_mask
