"""
Eph/Ephrin Signaling Model

Comprehensive model of Eph receptor and Ephrin ligand signaling dynamics,
including forward and reverse signaling pathways, receptor clustering,
and downstream effector activation.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class EphEphrinParameters:
    """Parameters for Eph/Ephrin signaling model"""
    
    # Receptor and ligand synthesis
    k_eph_synth: float = 0.5  # Eph receptor synthesis rate
    k_ephrin_synth: float = 0.5  # Ephrin ligand synthesis rate
    
    # Degradation rates
    k_eph_deg: float = 0.1  # Eph receptor degradation
    k_ephrin_deg: float = 0.1  # Ephrin ligand degradation
    
    # Binding kinetics
    k_bind: float = 1.0  # Eph-Ephrin binding rate
    k_unbind: float = 0.1  # Eph-Ephrin unbinding rate
    
    # Clustering and activation
    k_cluster: float = 0.5  # Receptor clustering rate
    k_decluster: float = 0.2  # Cluster dissociation
    k_activation: float = 2.0  # Kinase activation rate
    k_deactivation: float = 0.5  # Deactivation rate
    
    # Forward signaling (into Eph-expressing cell)
    k_forward_max: float = 3.0  # Maximum forward signaling
    K_forward: float = 10.0  # Michaelis constant for forward signaling
    
    # Reverse signaling (into Ephrin-expressing cell)
    k_reverse_max: float = 2.0  # Maximum reverse signaling
    K_reverse: float = 8.0  # Michaelis constant for reverse signaling
    
    # Downstream effectors
    k_rac_act: float = 1.5  # Rac1 activation (attractive)
    k_rac_inact: float = 0.3  # Rac1 inactivation
    k_rho_act: float = 2.0  # RhoA activation (repulsive)
    k_rho_inact: float = 0.4  # RhoA inactivation
    
    # Cross-talk
    k_rac_rho_inhib: float = 0.5  # Rac inhibits Rho
    k_rho_rac_inhib: float = 0.5  # Rho inhibits Rac
    
    # Endocytosis
    k_endocytosis: float = 0.3  # Complex internalization
    k_recycling: float = 0.1  # Receptor recycling


class EphEphrinSignaling:
    """
    Comprehensive Eph/Ephrin signaling model with bidirectional signaling
    """
    
    def __init__(self, params: EphEphrinParameters = None):
        self.params = params or EphEphrinParameters()
        
    def derivatives(self, state: np.ndarray, t: float, 
                   external_ephrin: float = 0.0,
                   external_eph: float = 0.0) -> np.ndarray:
        """
        Calculate derivatives for the Eph/Ephrin signaling system
        
        State variables:
        0: Eph (free receptor)
        1: Ephrin (free ligand)
        2: Eph_Ephrin (bound complex)
        3: Eph_Ephrin_cluster (clustered complexes)
        4: Eph_active (activated receptor)
        5: Forward_signal (signal into Eph cell)
        6: Reverse_signal (signal into Ephrin cell)
        7: Rac_GTP (active Rac1)
        8: RhoA_GTP (active RhoA)
        9: Eph_internal (internalized receptor)
        """
        
        Eph, Ephrin, Complex, Cluster, Eph_act, Fwd, Rev, Rac, Rho, Eph_int = state
        p = self.params
        
        # Eph-Ephrin binding/unbinding
        binding_rate = p.k_bind * Eph * (Ephrin + external_ephrin)
        unbinding_rate = p.k_unbind * Complex
        
        # Clustering
        clustering_rate = p.k_cluster * Complex * Complex
        declustering_rate = p.k_decluster * Cluster
        
        # Receptor activation (requires clustering)
        activation_rate = p.k_activation * Cluster
        deactivation_rate = p.k_deactivation * Eph_act
        
        # Forward signaling (into Eph-expressing cell)
        forward_signal = (p.k_forward_max * Eph_act) / (p.K_forward + Eph_act)
        forward_decay = 0.5 * Fwd
        
        # Reverse signaling (into Ephrin-expressing cell)
        reverse_signal = (p.k_reverse_max * Cluster) / (p.K_reverse + Cluster)
        reverse_decay = 0.5 * Rev
        
        # Rac/Rho dynamics with cross-inhibition
        rac_activation = p.k_rac_act * Fwd / (1 + p.k_rho_rac_inhib * Rho)
        rac_inactivation = p.k_rac_inact * Rac
        
        rho_activation = p.k_rho_act * Fwd / (1 + p.k_rac_rho_inhib * Rac)
        rho_inactivation = p.k_rho_inact * Rho
        
        # Endocytosis
        endocytosis_rate = p.k_endocytosis * Cluster
        recycling_rate = p.k_recycling * Eph_int
        
        # ODEs
        dEph = (p.k_eph_synth - p.k_eph_deg * Eph 
                - binding_rate + unbinding_rate + recycling_rate)
        
        dEphrin = (p.k_ephrin_synth - p.k_ephrin_deg * Ephrin 
                   - binding_rate + unbinding_rate)
        
        dComplex = (binding_rate - unbinding_rate 
                    - 2 * clustering_rate + declustering_rate)
        
        dCluster = (clustering_rate - declustering_rate - endocytosis_rate)
        
        dEph_act = activation_rate - deactivation_rate
        
        dFwd = forward_signal - forward_decay
        
        dRev = reverse_signal - reverse_decay
        
        dRac = rac_activation - rac_inactivation
        
        dRho = rho_activation - rho_inactivation
        
        dEph_int = endocytosis_rate - recycling_rate - 0.05 * Eph_int
        
        return np.array([dEph, dEphrin, dComplex, dCluster, dEph_act, 
                        dFwd, dRev, dRac, dRho, dEph_int])
    
    def simulate(self, t_span: Tuple[float, float], n_points: int = 1000,
                 initial_state: np.ndarray = None,
                 external_ephrin: float = 0.0,
                 external_eph: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Eph/Ephrin signaling system
        
        Args:
            t_span: Time span (start, end)
            n_points: Number of time points
            initial_state: Initial conditions
            external_ephrin: External ephrin concentration
            external_eph: External eph concentration
            
        Returns:
            Time array and state array
        """
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        if initial_state is None:
            # Default initial conditions
            initial_state = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 
                                     0.0, 0.0, 1.0, 1.0, 0.0])
        
        solution = odeint(self.derivatives, initial_state, t,
                         args=(external_ephrin, external_eph))
        
        return t, solution
    
    def calculate_repulsion_strength(self, state: np.ndarray) -> float:
        """
        Calculate net repulsion strength based on Rho/Rac balance
        
        Positive values = repulsion (RhoA dominant)
        Negative values = attraction (Rac dominant)
        """
        rac = state[7]
        rho = state[8]
        return rho - rac
    
    def get_state_dict(self, state: np.ndarray) -> Dict[str, float]:
        """Convert state vector to labeled dictionary"""
        return {
            'Eph': state[0],
            'Ephrin': state[1],
            'Complex': state[2],
            'Cluster': state[3],
            'Eph_active': state[4],
            'Forward_signal': state[5],
            'Reverse_signal': state[6],
            'Rac_GTP': state[7],
            'RhoA_GTP': state[8],
            'Eph_internal': state[9]
        }


def analyze_gradient_response(gradient_levels: np.ndarray,
                              params: EphEphrinParameters = None) -> np.ndarray:
    """
    Analyze how cells respond to different ephrin gradient levels
    
    Args:
        gradient_levels: Array of ephrin concentrations
        params: Model parameters
        
    Returns:
        Array of repulsion strengths at each gradient level
    """
    model = EphEphrinSignaling(params)
    repulsion_strengths = []
    
    for ephrin_level in gradient_levels:
        t, solution = model.simulate(
            (0, 50), 
            n_points=500,
            external_ephrin=ephrin_level
        )
        final_state = solution[-1]
        repulsion = model.calculate_repulsion_strength(final_state)
        repulsion_strengths.append(repulsion)
    
    return np.array(repulsion_strengths)


def simulate_ephrin_pulse(pulse_time: float = 10.0, 
                          pulse_duration: float = 5.0,
                          pulse_amplitude: float = 10.0,
                          total_time: float = 50.0,
                          params: EphEphrinParameters = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate response to a pulse of ephrin stimulation
    
    Returns:
        Time array, solution array, ephrin input array
    """
    model = EphEphrinSignaling(params)
    n_points = 1000
    t = np.linspace(0, total_time, n_points)
    
    # Create ephrin pulse
    ephrin_input = np.zeros_like(t)
    pulse_mask = (t >= pulse_time) & (t < pulse_time + pulse_duration)
    ephrin_input[pulse_mask] = pulse_amplitude
    
    # Simulate with time-varying input
    initial_state = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 1.0, 1.0, 0.0])
    
    solution = []
    state = initial_state
    
    for i in range(len(t)):
        if i > 0:
            dt = t[i] - t[i-1]
            deriv = model.derivatives(state, t[i], 
                                     external_ephrin=ephrin_input[i])
            state = state + deriv * dt
            state = np.maximum(state, 0)  # Prevent negative concentrations
        solution.append(state.copy())
    
    return t, np.array(solution), ephrin_input
