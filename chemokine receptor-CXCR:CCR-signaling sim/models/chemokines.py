"""
chemokine ligand models and gradient generation
implements spatial distribution and diffusion dynamics
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import scipy.ndimage as ndimage


@dataclass
class ChemokineProperties:
    """biophysical properties of chemokine ligands"""
    name: str
    family: str  # CXC, CC, C, CX3C
    molecular_weight: float  # kDa
    diffusion_coeff: float  # μm²/s
    degradation_rate: float  # s^-1
    production_rate: float  # molecules/s per source cell
    receptor_targets: list  # list of receptor names


class ChemokineGradient:
    """
    spatial chemokine distribution with diffusion and degradation
    uses finite difference methods for pde integration
    """
    
    def __init__(self, 
                 properties: ChemokineProperties,
                 grid_size: Tuple[int, int, int],
                 grid_spacing: float):
        """
        args:
            properties: chemokine properties
            grid_size: (nx, ny, nz) grid dimensions
            grid_spacing: distance between grid points (μm)
        """
        self.props = properties
        self.grid_size = grid_size
        self.dx = grid_spacing
        
        # concentration field (molecules/μm³)
        self.concentration = np.zeros(grid_size)
        
        # source field (production sites)
        self.sources = np.zeros(grid_size)
        
        # boundary conditions
        self.boundary_value = 0.0
        
    def add_point_source(self, position: Tuple[int, int, int], 
                        strength: float):
        """
        add chemokine source at position
        
        args:
            position: (i, j, k) grid indices
            strength: production rate multiplier
        """
        self.sources[position] = strength * self.props.production_rate
    
    def add_sphere_source(self, center: Tuple[int, int, int], 
                         radius: int, strength: float):
        """
        add spherical source region (e.g., inflamed tissue)
        """
        nx, ny, nz = self.grid_size
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        distance = np.sqrt((X - center[0])**2 + 
                          (Y - center[1])**2 + 
                          (Z - center[2])**2)
        
        mask = distance <= radius
        self.sources[mask] = strength * self.props.production_rate
    
    def add_plane_source(self, axis: int, position: int, 
                        thickness: int, strength: float):
        """
        add planar source (e.g., endothelial layer)
        
        args:
            axis: 0, 1, or 2 for x, y, z
            position: location along axis
            thickness: source thickness in grid points
        """
        if axis == 0:
            self.sources[position:position+thickness, :, :] = \
                strength * self.props.production_rate
        elif axis == 1:
            self.sources[:, position:position+thickness, :] = \
                strength * self.props.production_rate
        else:
            self.sources[:, :, position:position+thickness] = \
                strength * self.props.production_rate
    
    def diffuse(self, dt: float):
        """
        3d diffusion using explicit finite difference
        
        ∂C/∂t = D∇²C
        """
        D = self.props.diffusion_coeff  # μm²/s
        dx2 = self.dx ** 2
        
        # diffusion coefficient for numerical stability
        alpha = D * dt / dx2
        
        # check stability (cfl condition: alpha ≤ 1/6 for 3d)
        if alpha > 1/6:
            raise ValueError(f"unstable diffusion: reduce dt or increase dx")
        
        # laplacian using convolution (6-point stencil)
        laplacian = (
            ndimage.laplace(self.concentration) / dx2
        )
        
        self.concentration += D * dt * laplacian
    
    def degrade(self, dt: float):
        """
        first-order degradation kinetics
        
        ∂C/∂t = -k*C
        """
        decay = np.exp(-self.props.degradation_rate * dt)
        self.concentration *= decay
    
    def produce(self, dt: float):
        """
        add production from sources
        """
        self.concentration += self.sources * dt
    
    def apply_boundaries(self):
        """
        apply boundary conditions (dirichlet)
        """
        self.concentration[0, :, :] = self.boundary_value
        self.concentration[-1, :, :] = self.boundary_value
        self.concentration[:, 0, :] = self.boundary_value
        self.concentration[:, -1, :] = self.boundary_value
        self.concentration[:, :, 0] = self.boundary_value
        self.concentration[:, :, -1] = self.boundary_value
    
    def step(self, dt: float):
        """
        integrate gradient dynamics for one time step
        """
        self.produce(dt)
        self.diffuse(dt)
        self.degrade(dt)
        self.apply_boundaries()
        
        # ensure non-negative
        self.concentration = np.maximum(self.concentration, 0)
    
    def get_concentration(self, position: Tuple[float, float, float]) -> float:
        """
        get concentration at continuous position using trilinear interpolation
        
        args:
            position: (x, y, z) in μm
        
        returns:
            concentration in molecules/μm³
        """
        # convert to grid coordinates
        i = position[0] / self.dx
        j = position[1] / self.dx
        k = position[2] / self.dx
        
        # bounds checking
        i = np.clip(i, 0, self.grid_size[0] - 1.001)
        j = np.clip(j, 0, self.grid_size[1] - 1.001)
        k = np.clip(k, 0, self.grid_size[2] - 1.001)
        
        # trilinear interpolation
        i0, i1 = int(np.floor(i)), int(np.ceil(i))
        j0, j1 = int(np.floor(j)), int(np.ceil(j))
        k0, k1 = int(np.floor(k)), int(np.ceil(k))
        
        # fractional parts
        fi = i - i0
        fj = j - j0
        fk = k - k0
        
        # interpolate
        c000 = self.concentration[i0, j0, k0]
        c001 = self.concentration[i0, j0, k1]
        c010 = self.concentration[i0, j1, k0]
        c011 = self.concentration[i0, j1, k1]
        c100 = self.concentration[i1, j0, k0]
        c101 = self.concentration[i1, j0, k1]
        c110 = self.concentration[i1, j1, k0]
        c111 = self.concentration[i1, j1, k1]
        
        c00 = c000 * (1 - fk) + c001 * fk
        c01 = c010 * (1 - fk) + c011 * fk
        c10 = c100 * (1 - fk) + c101 * fk
        c11 = c110 * (1 - fk) + c111 * fk
        
        c0 = c00 * (1 - fj) + c01 * fj
        c1 = c10 * (1 - fj) + c11 * fj
        
        return c0 * (1 - fi) + c1 * fi
    
    def get_gradient(self, position: Tuple[float, float, float]) -> np.ndarray:
        """
        compute concentration gradient at position
        
        returns:
            gradient vector (∂C/∂x, ∂C/∂y, ∂C/∂z) in molecules/μm⁴
        """
        # convert to grid coordinates
        i = int(position[0] / self.dx)
        j = int(position[1] / self.dx)
        k = int(position[2] / self.dx)
        
        # bounds
        i = np.clip(i, 1, self.grid_size[0] - 2)
        j = np.clip(j, 1, self.grid_size[1] - 2)
        k = np.clip(k, 1, self.grid_size[2] - 2)
        
        # central differences
        grad_x = (self.concentration[i+1, j, k] - 
                 self.concentration[i-1, j, k]) / (2 * self.dx)
        grad_y = (self.concentration[i, j+1, k] - 
                 self.concentration[i, j-1, k]) / (2 * self.dx)
        grad_z = (self.concentration[i, j, k+1] - 
                 self.concentration[i, j, k-1]) / (2 * self.dx)
        
        return np.array([grad_x, grad_y, grad_z])
    
    def to_molar(self, molecules_per_um3: float) -> float:
        """
        convert molecules/μm³ to molar concentration
        
        1 μm³ = 1e-15 L
        1 M = 6.022e23 molecules/L
        """
        avogadro = 6.022e23
        molecules_per_L = molecules_per_um3 * 1e15
        return molecules_per_L / avogadro


# chemokine library with literature values

CXCL8 = ChemokineProperties(
    name="CXCL8 (IL-8)",
    family="CXC",
    molecular_weight=8.0,  # kDa
    diffusion_coeff=100.0,  # μm²/s
    degradation_rate=0.001,  # s^-1 (half-life ~11 min)
    production_rate=1e6,  # molecules/s per cell
    receptor_targets=["CXCR1", "CXCR2"]
)

CXCL12 = ChemokineProperties(
    name="CXCL12 (SDF-1)",
    family="CXC",
    molecular_weight=8.0,
    diffusion_coeff=90.0,
    degradation_rate=0.0005,  # longer half-life
    production_rate=8e5,
    receptor_targets=["CXCR4"]
)

CCL2 = ChemokineProperties(
    name="CCL2 (MCP-1)",
    family="CC",
    molecular_weight=8.6,
    diffusion_coeff=85.0,
    degradation_rate=0.0008,
    production_rate=1.2e6,
    receptor_targets=["CCR2"]
)

CCL5 = ChemokineProperties(
    name="CCL5 (RANTES)",
    family="CC",
    molecular_weight=7.8,
    diffusion_coeff=95.0,
    degradation_rate=0.0006,
    production_rate=9e5,
    receptor_targets=["CCR1", "CCR5"]
)

CCL19 = ChemokineProperties(
    name="CCL19 (ELC)",
    family="CC",
    molecular_weight=8.0,
    diffusion_coeff=88.0,
    degradation_rate=0.0004,
    production_rate=7e5,
    receptor_targets=["CCR7"]
)

CCL21 = ChemokineProperties(
    name="CCL21 (SLC)",
    family="CC",
    molecular_weight=8.4,
    diffusion_coeff=82.0,
    degradation_rate=0.0003,  # very stable
    production_rate=6e5,
    receptor_targets=["CCR7"]
)


CHEMOKINE_LIBRARY = {
    "CXCL8": CXCL8,
    "CXCL12": CXCL12,
    "CCL2": CCL2,
    "CCL5": CCL5,
    "CCL19": CCL19,
    "CCL21": CCL21
}


def create_inflammation_gradient(grid_size: Tuple[int, int, int],
                                 grid_spacing: float) -> ChemokineGradient:
    """
    create typical inflammatory chemokine gradient (cxcl8)
    point source representing infection site
    """
    gradient = ChemokineGradient(CXCL8, grid_size, grid_spacing)
    
    # infection site at center bottom
    center = (grid_size[0] // 2, grid_size[1] // 2, 5)
    gradient.add_sphere_source(center, radius=3, strength=10.0)
    
    return gradient


def create_lymph_node_gradient(grid_size: Tuple[int, int, int],
                               grid_spacing: float) -> ChemokineGradient:
    """
    create homeostatic ccl21 gradient (lymph node homing)
    """
    gradient = ChemokineGradient(CCL21, grid_size, grid_spacing)
    
    # lymphatic endothelium source plane
    gradient.add_plane_source(axis=2, position=0, thickness=2, strength=5.0)
    
    return gradient
