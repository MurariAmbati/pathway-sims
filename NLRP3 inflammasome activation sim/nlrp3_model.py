"""
NLRP3 Inflammasome Activation Pathway Model

Simulates the two-signal model of NLRP3 inflammasome activation:
- Signal 1 (Priming): NF-κB-mediated transcription of pro-IL-1β, pro-IL-18, NLRP3
- Signal 2 (Activation): NLRP3 oligomerization, ASC recruitment, caspase-1 activation
- Outcomes: IL-1β/IL-18 maturation, GSDMD cleavage, pyroptosis
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class Parameters:
    """Model parameters for NLRP3 inflammasome pathway"""
    
    # Signal 1: Priming (NF-κB pathway)
    k_nfkb_act: float = 0.5      # NF-κB activation rate
    k_nfkb_deact: float = 0.1    # NF-κB deactivation rate
    k_pro_il1b: float = 0.8      # Pro-IL-1β synthesis rate
    k_pro_il18: float = 0.7      # Pro-IL-18 synthesis rate
    k_nlrp3_syn: float = 0.6     # NLRP3 synthesis rate
    
    # Signal 2: Activation triggers
    k_trigger: float = 1.0       # Trigger signal strength (K+ efflux, ROS, etc.)
    k_nlrp3_act: float = 0.7     # NLRP3 activation rate
    k_nlrp3_deact: float = 0.05  # NLRP3 deactivation rate
    
    # Inflammasome assembly
    k_asc_recruit: float = 1.2   # ASC recruitment rate
    k_casp1_recruit: float = 1.0 # Pro-caspase-1 recruitment rate
    k_inflamm_form: float = 0.9  # Inflammasome complex formation
    
    # Caspase-1 activation
    k_casp1_act: float = 1.5     # Caspase-1 auto-cleavage rate
    k_casp1_degrad: float = 0.2  # Active caspase-1 degradation
    
    # IL-1β and IL-18 maturation
    k_il1b_mat: float = 2.0      # IL-1β maturation by caspase-1
    k_il18_mat: float = 1.8      # IL-18 maturation by caspase-1
    k_cytok_secret: float = 0.5  # Cytokine secretion rate
    
    # GSDMD cleavage and pyroptosis
    k_gsdmd_cleave: float = 1.5  # GSDMD cleavage by caspase-1
    k_pore_form: float = 1.0     # Pore formation rate
    k_pyroptosis: float = 0.8    # Pyroptotic cell death rate
    
    # Degradation rates
    d_protein: float = 0.1       # General protein degradation
    d_cytokine: float = 0.15     # Cytokine degradation
    d_complex: float = 0.05      # Complex degradation


class NLRP3Model:
    """NLRP3 inflammasome pathway simulation model"""
    
    def __init__(self, params: Parameters = None):
        self.params = params if params else Parameters()
        self.state_names = [
            'NF-κB (active)',
            'Pro-IL-1β',
            'Pro-IL-18',
            'NLRP3',
            'NLRP3* (active)',
            'ASC',
            'Pro-caspase-1',
            'Inflammasome',
            'Caspase-1 (active)',
            'IL-1β (mature)',
            'IL-18 (mature)',
            'GSDMD',
            'GSDMD-NT',
            'Membrane pores',
            'Cell viability'
        ]
        
    def derivatives(self, state: np.ndarray, t: float, 
                   signal1: float, signal2: float) -> np.ndarray:
        """
        Calculate derivatives for NLRP3 inflammasome pathway
        
        Args:
            state: Current state vector
            t: Time point
            signal1: Priming signal strength (0-1)
            signal2: Activation signal strength (0-1)
        
        Returns:
            Array of derivatives
        """
        # Unpack state variables
        (nfkb, pro_il1b, pro_il18, nlrp3, nlrp3_act, asc, pro_casp1,
         inflamm, casp1_act, il1b, il18, gsdmd, gsdmd_nt, pores, viability) = state
        
        p = self.params
        
        # Signal 1: NF-κB activation (priming)
        d_nfkb = (p.k_nfkb_act * signal1 * (1 - nfkb) - 
                  p.k_nfkb_deact * nfkb)
        
        # Pro-inflammatory gene transcription
        d_pro_il1b = p.k_pro_il1b * nfkb - p.d_protein * pro_il1b - p.k_il1b_mat * casp1_act * pro_il1b
        d_pro_il18 = p.k_pro_il18 * nfkb - p.d_protein * pro_il18 - p.k_il18_mat * casp1_act * pro_il18
        d_nlrp3 = p.k_nlrp3_syn * nfkb - p.d_protein * nlrp3 - p.k_nlrp3_act * signal2 * nlrp3
        
        # Signal 2: NLRP3 activation
        d_nlrp3_act = (p.k_nlrp3_act * signal2 * nlrp3 - 
                       p.k_nlrp3_deact * nlrp3_act -
                       p.k_inflamm_form * nlrp3_act * asc * pro_casp1)
        
        # ASC and pro-caspase-1 dynamics
        d_asc = (-p.k_asc_recruit * nlrp3_act * asc * (1 - inflamm) -
                 p.k_inflamm_form * nlrp3_act * asc * pro_casp1)
        
        d_pro_casp1 = (-p.k_casp1_recruit * inflamm * pro_casp1 -
                       p.k_inflamm_form * nlrp3_act * asc * pro_casp1)
        
        # Inflammasome complex formation
        d_inflamm = (p.k_inflamm_form * nlrp3_act * asc * pro_casp1 -
                     p.d_complex * inflamm)
        
        # Caspase-1 activation
        d_casp1_act = (p.k_casp1_act * inflamm * (1 + casp1_act * 0.5) -  # Autocatalytic
                       p.k_casp1_degrad * casp1_act)
        
        # IL-1β and IL-18 maturation
        d_il1b = (p.k_il1b_mat * casp1_act * pro_il1b -
                  p.k_cytok_secret * il1b -
                  p.d_cytokine * il1b)
        
        d_il18 = (p.k_il18_mat * casp1_act * pro_il18 -
                  p.k_cytok_secret * il18 -
                  p.d_cytokine * il18)
        
        # GSDMD cleavage and pore formation
        d_gsdmd = -p.k_gsdmd_cleave * casp1_act * gsdmd
        
        d_gsdmd_nt = (p.k_gsdmd_cleave * casp1_act * gsdmd -
                      p.k_pore_form * gsdmd_nt)
        
        d_pores = p.k_pore_form * gsdmd_nt - p.d_protein * pores
        
        # Pyroptotic cell death
        d_viability = -p.k_pyroptosis * pores * viability
        
        return np.array([
            d_nfkb, d_pro_il1b, d_pro_il18, d_nlrp3, d_nlrp3_act,
            d_asc, d_pro_casp1, d_inflamm, d_casp1_act,
            d_il1b, d_il18, d_gsdmd, d_gsdmd_nt, d_pores, d_viability
        ])
    
    def simulate(self, t_span: Tuple[float, float], 
                signal1_profile: callable,
                signal2_profile: callable,
                initial_state: np.ndarray = None,
                n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the NLRP3 inflammasome pathway
        
        Args:
            t_span: (start_time, end_time) in minutes
            signal1_profile: Function t -> signal1 strength
            signal2_profile: Function t -> signal2 strength
            initial_state: Initial conditions (optional)
            n_points: Number of time points
        
        Returns:
            (time_points, states) arrays
        """
        if initial_state is None:
            # Resting state
            initial_state = np.array([
                0.0,    # NF-κB (inactive)
                0.1,    # Pro-IL-1β (basal)
                0.1,    # Pro-IL-18 (basal)
                0.2,    # NLRP3 (basal)
                0.0,    # NLRP3* (inactive)
                1.0,    # ASC (abundant)
                1.0,    # Pro-caspase-1 (abundant)
                0.0,    # Inflammasome (none)
                0.0,    # Caspase-1 (inactive)
                0.0,    # IL-1β (none)
                0.0,    # IL-18 (none)
                1.0,    # GSDMD (intact)
                0.0,    # GSDMD-NT (none)
                0.0,    # Membrane pores (none)
                1.0     # Cell viability (100%)
            ])
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        # Solve ODE system
        states = odeint(
            lambda y, t: self.derivatives(
                y, t, 
                signal1_profile(t),
                signal2_profile(t)
            ),
            initial_state,
            t
        )
        
        return t, states
    
    def get_state_dict(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert state array to dictionary with named components"""
        return {name: states[:, i] for i, name in enumerate(self.state_names)}


# Signal profile generators
def step_signal(t_start: float, amplitude: float = 1.0):
    """Generate step function signal"""
    return lambda t: amplitude if t >= t_start else 0.0


def pulse_signal(t_start: float, duration: float, amplitude: float = 1.0):
    """Generate pulse signal"""
    return lambda t: amplitude if t_start <= t < t_start + duration else 0.0


def ramp_signal(t_start: float, t_end: float, amplitude: float = 1.0):
    """Generate ramping signal"""
    def signal(t):
        if t < t_start:
            return 0.0
        elif t > t_end:
            return amplitude
        else:
            return amplitude * (t - t_start) / (t_end - t_start)
    return signal


def oscillating_signal(period: float, amplitude: float = 1.0, phase: float = 0.0):
    """Generate oscillating signal"""
    return lambda t: amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * t / period + phase))
