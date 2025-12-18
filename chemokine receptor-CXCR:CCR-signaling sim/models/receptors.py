"""
chemokine receptor models (cxcr/ccr families)
implements g-protein coupled receptor dynamics and binding kinetics
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum


class ReceptorFamily(Enum):
    """chemokine receptor classification"""
    CXCR = "cxcr"
    CCR = "ccr"
    XCR = "xcr"
    CX3CR = "cx3cr"


@dataclass
class ReceptorProperties:
    """biophysical properties of chemokine receptors"""
    name: str
    family: ReceptorFamily
    kon: float  # association rate constant (M^-1 s^-1)
    koff: float  # dissociation rate constant (s^-1)
    density: int  # receptors per cell
    desensitization_rate: float  # s^-1
    internalization_rate: float  # s^-1
    recycling_rate: float  # s^-1
    
    @property
    def kd(self) -> float:
        """equilibrium dissociation constant (M)"""
        return self.koff / self.kon
    
    @property
    def kd_nm(self) -> float:
        """kd in nanomolar"""
        return self.kd * 1e9


class ChemokineReceptor:
    """
    base class for chemokine receptor signaling
    models receptor states: free, bound, desensitized, internalized
    """
    
    def __init__(self, properties: ReceptorProperties):
        self.props = properties
        
        # receptor state populations
        self.R_free = properties.density  # free receptors
        self.R_bound = 0.0  # ligand-bound receptors
        self.R_desensitized = 0.0  # desensitized receptors
        self.R_internalized = 0.0  # internalized receptors
        
        # g-protein coupling state
        self.g_protein_active = 0.0
        
        # signal amplification
        self.signal_amplitude = 0.0
        
    def bind_ligand(self, ligand_conc: float, dt: float) -> float:
        """
        receptor-ligand binding kinetics
        
        R + L <-> RL
        d[RL]/dt = kon*[R]*[L] - koff*[RL]
        
        args:
            ligand_conc: chemokine concentration (M)
            dt: time step (s)
        
        returns:
            newly bound receptors
        """
        # forward binding
        binding = self.props.kon * self.R_free * ligand_conc * dt
        
        # reverse dissociation
        dissociation = self.props.koff * self.R_bound * dt
        
        # update populations
        delta = binding - dissociation
        self.R_free -= delta
        self.R_bound += delta
        
        return delta
    
    def desensitize(self, dt: float):
        """
        receptor desensitization (grk phosphorylation, arrestin binding)
        """
        desensitized = self.props.desensitization_rate * self.R_bound * dt
        self.R_bound -= desensitized
        self.R_desensitized += desensitized
    
    def internalize(self, dt: float):
        """
        receptor internalization via clathrin-mediated endocytosis
        """
        internalized = self.props.internalization_rate * self.R_desensitized * dt
        self.R_desensitized -= internalized
        self.R_internalized += internalized
    
    def recycle(self, dt: float):
        """
        receptor recycling back to membrane
        """
        recycled = self.props.recycling_rate * self.R_internalized * dt
        self.R_internalized -= recycled
        self.R_free += recycled
    
    def activate_g_protein(self, dt: float, deactivation_rate: float = 0.5):
        """
        g-protein activation by bound receptors
        each bound receptor can activate multiple g-proteins (amplification)
        """
        amplification_factor = 10.0  # g-proteins per receptor per second
        
        activation = amplification_factor * self.R_bound * dt
        deactivation = deactivation_rate * self.g_protein_active * dt
        
        self.g_protein_active += activation - deactivation
        self.signal_amplitude = self.g_protein_active
    
    def step(self, ligand_conc: float, dt: float):
        """
        integrate receptor dynamics for one time step
        """
        self.bind_ligand(ligand_conc, dt)
        self.desensitize(dt)
        self.internalize(dt)
        self.recycle(dt)
        self.activate_g_protein(dt)
    
    def occupancy(self) -> float:
        """fraction of receptors bound to ligand"""
        total = self.R_free + self.R_bound + self.R_desensitized + self.R_internalized
        return self.R_bound / max(total, 1.0)
    
    def get_state(self) -> Dict[str, float]:
        """return current receptor state"""
        return {
            'free': self.R_free,
            'bound': self.R_bound,
            'desensitized': self.R_desensitized,
            'internalized': self.R_internalized,
            'g_protein_active': self.g_protein_active,
            'occupancy': self.occupancy(),
            'signal': self.signal_amplitude
        }


# predefined receptor types with literature values
CXCR1 = ReceptorProperties(
    name="CXCR1",
    family=ReceptorFamily.CXCR,
    kon=1e7,  # M^-1 s^-1
    koff=0.1,  # s^-1, kd ~ 10 nM
    density=50000,
    desensitization_rate=0.05,  # s^-1
    internalization_rate=0.01,
    recycling_rate=0.005
)

CXCR2 = ReceptorProperties(
    name="CXCR2",
    family=ReceptorFamily.CXCR,
    kon=5e6,
    koff=0.05,  # kd ~ 10 nM
    density=60000,
    desensitization_rate=0.08,
    internalization_rate=0.015,
    recycling_rate=0.006
)

CXCR4 = ReceptorProperties(
    name="CXCR4",
    family=ReceptorFamily.CXCR,
    kon=2e7,
    koff=0.02,  # kd ~ 1 nM (high affinity)
    density=40000,
    desensitization_rate=0.04,
    internalization_rate=0.012,
    recycling_rate=0.004
)

CCR1 = ReceptorProperties(
    name="CCR1",
    family=ReceptorFamily.CCR,
    kon=8e6,
    koff=0.08,  # kd ~ 10 nM
    density=30000,
    desensitization_rate=0.06,
    internalization_rate=0.01,
    recycling_rate=0.005
)

CCR2 = ReceptorProperties(
    name="CCR2",
    family=ReceptorFamily.CCR,
    kon=1e7,
    koff=0.15,  # kd ~ 15 nM
    density=45000,
    desensitization_rate=0.07,
    internalization_rate=0.013,
    recycling_rate=0.006
)

CCR5 = ReceptorProperties(
    name="CCR5",
    family=ReceptorFamily.CCR,
    kon=1.5e7,
    koff=0.03,  # kd ~ 2 nM
    density=35000,
    desensitization_rate=0.05,
    internalization_rate=0.011,
    recycling_rate=0.0045
)

CCR7 = ReceptorProperties(
    name="CCR7",
    family=ReceptorFamily.CCR,
    kon=1.2e7,
    koff=0.12,  # kd ~ 10 nM
    density=25000,
    desensitization_rate=0.045,
    internalization_rate=0.009,
    recycling_rate=0.004
)


# receptor registry
RECEPTOR_LIBRARY = {
    "CXCR1": CXCR1,
    "CXCR2": CXCR2,
    "CXCR4": CXCR4,
    "CCR1": CCR1,
    "CCR2": CCR2,
    "CCR5": CCR5,
    "CCR7": CCR7
}


class ReceptorExpression:
    """
    models cell-type specific receptor expression profiles
    """
    
    def __init__(self, cell_type: str):
        self.cell_type = cell_type
        self.receptors: Dict[str, ChemokineReceptor] = {}
        
    def express_receptor(self, receptor_name: str, 
                        density_modifier: float = 1.0):
        """
        add receptor to cell surface
        
        args:
            receptor_name: name from RECEPTOR_LIBRARY
            density_modifier: scaling factor for receptor density
        """
        if receptor_name in RECEPTOR_LIBRARY:
            props = RECEPTOR_LIBRARY[receptor_name]
            
            # modify density
            modified_props = ReceptorProperties(
                name=props.name,
                family=props.family,
                kon=props.kon,
                koff=props.koff,
                density=int(props.density * density_modifier),
                desensitization_rate=props.desensitization_rate,
                internalization_rate=props.internalization_rate,
                recycling_rate=props.recycling_rate
            )
            
            self.receptors[receptor_name] = ChemokineReceptor(modified_props)
    
    def get_total_signal(self) -> float:
        """sum signal across all expressed receptors"""
        return sum(r.signal_amplitude for r in self.receptors.values())
    
    def step_all(self, ligand_concentrations: Dict[str, float], dt: float):
        """
        update all receptors
        
        args:
            ligand_concentrations: dict mapping receptor name to ligand conc
            dt: time step
        """
        for receptor_name, receptor in self.receptors.items():
            ligand_conc = ligand_concentrations.get(receptor_name, 0.0)
            receptor.step(ligand_conc, dt)


def create_neutrophil_receptors() -> ReceptorExpression:
    """typical neutrophil receptor profile"""
    cell = ReceptorExpression("neutrophil")
    cell.express_receptor("CXCR1", 1.2)
    cell.express_receptor("CXCR2", 1.5)
    cell.express_receptor("CXCR4", 0.3)
    return cell


def create_t_cell_receptors() -> ReceptorExpression:
    """typical t cell receptor profile"""
    cell = ReceptorExpression("t_cell")
    cell.express_receptor("CXCR4", 1.0)
    cell.express_receptor("CCR7", 1.2)
    cell.express_receptor("CCR5", 0.4)
    return cell


def create_monocyte_receptors() -> ReceptorExpression:
    """typical monocyte receptor profile"""
    cell = ReceptorExpression("monocyte")
    cell.express_receptor("CCR1", 0.8)
    cell.express_receptor("CCR2", 1.3)
    cell.express_receptor("CCR5", 0.6)
    cell.express_receptor("CXCR4", 0.5)
    return cell
