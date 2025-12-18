"""
tissue microenvironment model
defines spatial domain and boundary conditions
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class TissueType(Enum):
    """tissue compartment types"""
    BLOOD_VESSEL = "blood_vessel"
    INTERSTITIUM = "interstitium"
    LYMPHATIC = "lymphatic"
    LYMPH_NODE = "lymph_node"
    BARRIER = "barrier"


@dataclass
class TissueGeometry:
    """defines spatial domain"""
    size: Tuple[float, float, float]  # (x, y, z) dimensions in μm
    grid_spacing: float  # μm per grid point
    
    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """grid dimensions"""
        return (
            int(self.size[0] / self.grid_spacing),
            int(self.size[1] / self.grid_spacing),
            int(self.size[2] / self.grid_spacing)
        )
    
    @property
    def volume(self) -> float:
        """tissue volume (μm³)"""
        return self.size[0] * self.size[1] * self.size[2]


class TissueCompartment:
    """
    spatial tissue model with multiple compartments
    """
    
    def __init__(self, geometry: TissueGeometry):
        self.geometry = geometry
        
        # compartment labels (grid)
        self.compartments = np.full(geometry.grid_size, 
                                    TissueType.INTERSTITIUM.value,
                                    dtype=object)
        
        # physical properties (grid-based)
        self.permeability = np.ones(geometry.grid_size)  # diffusion scaling
        self.adhesion_molecules = np.zeros(geometry.grid_size)  # selectins, integrins
        
    def define_blood_vessel(self, 
                           center: Tuple[int, int, int],
                           radius: int,
                           axis: int = 2):
        """
        define cylindrical blood vessel region
        
        args:
            center: vessel center grid coordinates
            radius: vessel radius in grid points
            axis: vessel axis (0=x, 1=y, 2=z)
        """
        nx, ny, nz = self.geometry.grid_size
        
        if axis == 2:  # z-axis vessel
            for i in range(nx):
                for j in range(ny):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if dist <= radius:
                        self.compartments[i, j, :] = TissueType.BLOOD_VESSEL.value
                        # high adhesion molecules on endothelium
                        if radius - 2 < dist <= radius:
                            self.adhesion_molecules[i, j, :] = 1.0
    
    def define_lymphatic(self,
                        position: Tuple[int, int],
                        thickness: int):
        """
        define lymphatic vessel layer
        
        args:
            position: (x, y) position
            thickness: vessel thickness in grid points
        """
        x, y = position
        self.compartments[x:x+thickness, y:y+thickness, :] = TissueType.LYMPHATIC.value
        self.permeability[x:x+thickness, y:y+thickness, :] = 2.0  # high permeability
    
    def define_barrier(self,
                      axis: int,
                      position: int,
                      thickness: int = 1):
        """
        define impermeable barrier (e.g., basement membrane)
        
        args:
            axis: 0, 1, or 2 for x, y, z
            position: location along axis
            thickness: barrier thickness
        """
        if axis == 0:
            self.compartments[position:position+thickness, :, :] = TissueType.BARRIER.value
            self.permeability[position:position+thickness, :, :] = 0.01
        elif axis == 1:
            self.compartments[:, position:position+thickness, :] = TissueType.BARRIER.value
            self.permeability[:, position:position+thickness, :] = 0.01
        else:
            self.compartments[:, :, position:position+thickness] = TissueType.BARRIER.value
            self.permeability[:, :, position:position+thickness] = 0.01
    
    def get_boundaries(self) -> Tuple[Tuple, Tuple, Tuple]:
        """
        return spatial boundaries for cell confinement
        """
        return (
            (0, self.geometry.size[0]),
            (0, self.geometry.size[1]),
            (0, self.geometry.size[2])
        )
    
    def is_in_compartment(self, 
                         position: np.ndarray,
                         compartment_type: TissueType) -> bool:
        """
        check if position is in specific compartment
        
        args:
            position: (x, y, z) in μm
            compartment_type: tissue type to check
        """
        # convert to grid coordinates
        i = int(position[0] / self.geometry.grid_spacing)
        j = int(position[1] / self.geometry.grid_spacing)
        k = int(position[2] / self.geometry.grid_spacing)
        
        # bounds check
        nx, ny, nz = self.geometry.grid_size
        i = np.clip(i, 0, nx - 1)
        j = np.clip(j, 0, ny - 1)
        k = np.clip(k, 0, nz - 1)
        
        return self.compartments[i, j, k] == compartment_type.value
    
    def get_adhesion_at_position(self, position: np.ndarray) -> float:
        """
        get adhesion molecule density at position
        """
        i = int(position[0] / self.geometry.grid_spacing)
        j = int(position[1] / self.geometry.grid_spacing)
        k = int(position[2] / self.geometry.grid_spacing)
        
        nx, ny, nz = self.geometry.grid_size
        i = np.clip(i, 0, nx - 1)
        j = np.clip(j, 0, ny - 1)
        k = np.clip(k, 0, nz - 1)
        
        return self.adhesion_molecules[i, j, k]


def create_inflammation_tissue(size: Tuple[float, float, float] = (200, 200, 100),
                               grid_spacing: float = 2.0) -> TissueCompartment:
    """
    create inflammatory tissue microenvironment
    
    - blood vessel at top
    - interstitial space
    - inflammation site at bottom
    
    args:
        size: tissue dimensions (μm)
        grid_spacing: grid resolution
    
    returns:
        TissueCompartment
    """
    geometry = TissueGeometry(size=size, grid_spacing=grid_spacing)
    tissue = TissueCompartment(geometry)
    
    # blood vessel at top
    center = (geometry.grid_size[0] // 2, geometry.grid_size[1] // 2, 0)
    tissue.define_blood_vessel(center, radius=10, axis=2)
    
    return tissue


def create_lymph_node_tissue(size: Tuple[float, float, float] = (300, 300, 200),
                             grid_spacing: float = 3.0) -> TissueCompartment:
    """
    create lymph node microenvironment
    
    - lymphatic vessels
    - t-cell zones
    - high adhesion regions
    """
    geometry = TissueGeometry(size=size, grid_spacing=grid_spacing)
    tissue = TissueCompartment(geometry)
    
    # lymphatic entry at boundary
    tissue.define_lymphatic((0, 0), thickness=5)
    
    # high adhesion in central region (high endothelial venules)
    center_x = geometry.grid_size[0] // 2
    center_y = geometry.grid_size[1] // 2
    tissue.adhesion_molecules[center_x-10:center_x+10, 
                            center_y-10:center_y+10, :] = 0.8
    
    return tissue


def create_simple_tissue(size: Tuple[float, float, float] = (150, 150, 150),
                        grid_spacing: float = 2.0) -> TissueCompartment:
    """
    create simple homogeneous tissue for basic simulations
    """
    geometry = TissueGeometry(size=size, grid_spacing=grid_spacing)
    tissue = TissueCompartment(geometry)
    
    return tissue
