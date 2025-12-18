"""
intracellular signaling pathway models
implements ode systems for signal transduction networks
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, Callable
from dataclasses import dataclass


@dataclass
class SignalingParameters:
    """kinetic parameters for signaling pathways"""
    # pi3k/akt pathway
    pi3k_activation_rate: float = 5.0  # s^-1
    akt_activation_rate: float = 8.0
    akt_deactivation_rate: float = 2.0
    
    # plc/calcium pathway
    plc_activation_rate: float = 6.0
    ip3_production_rate: float = 10.0
    ip3_degradation_rate: float = 0.5
    ca_release_rate: float = 50.0
    ca_reuptake_rate: float = 20.0
    
    # mapk/erk pathway
    mek_activation_rate: float = 4.0
    erk_activation_rate: float = 7.0
    erk_deactivation_rate: float = 1.5
    
    # small gtpases
    rac_activation_rate: float = 10.0
    rac_deactivation_rate: float = 3.0
    cdc42_activation_rate: float = 8.0
    cdc42_deactivation_rate: float = 2.5


class SignalingNetwork:
    """
    complete chemokine receptor signaling network
    integrates multiple pathways activated by g-protein coupling
    """
    
    def __init__(self, params: SignalingParameters = None):
        self.params = params or SignalingParameters()
        
        # pathway state variables
        self.state = {
            # g-protein signaling
            'g_alpha_gtp': 0.0,
            'g_betagamma': 0.0,
            
            # pi3k/akt pathway
            'pi3k_active': 0.0,
            'pip3': 0.0,
            'akt_active': 0.0,
            
            # plc/calcium pathway
            'plc_active': 0.0,
            'ip3': 0.0,
            'dag': 0.0,
            'ca_cyto': 0.1,  # μM baseline
            
            # mapk pathway
            'mek_active': 0.0,
            'erk_active': 0.0,
            
            # small gtpases
            'rac_gtp': 0.0,
            'cdc42_gtp': 0.0,
            'rho_gtp': 0.0
        }
        
        # cellular outputs
        self.outputs = {
            'migration_signal': 0.0,
            'adhesion_signal': 0.0,
            'transcription_signal': 0.0,
            'degranulation_signal': 0.0
        }
    
    def ode_system(self, state: np.ndarray, t: float, 
                   g_protein_input: float) -> np.ndarray:
        """
        ode system for signaling network
        
        args:
            state: vector of state variables
            t: time (not used, autonomous system with input)
            g_protein_input: activated g-protein level from receptors
        
        returns:
            derivatives dstate/dt
        """
        # unpack state
        (g_alpha, g_bg, pi3k, pip3, akt,
         plc, ip3, dag, ca, mek, erk,
         rac, cdc42, rho) = state
        
        p = self.params
        
        # g-protein dynamics
        dg_alpha = g_protein_input - 0.5 * g_alpha  # gtp hydrolysis
        dg_bg = g_protein_input - 0.3 * g_bg  # reassociation
        
        # pi3k/akt pathway
        dpi3k = p.pi3k_activation_rate * g_bg - 1.0 * pi3k
        dpip3 = 5.0 * pi3k - 0.8 * pip3  # pip2 -> pip3
        dakt = p.akt_activation_rate * pip3 - p.akt_deactivation_rate * akt
        
        # plc/calcium pathway
        dplc = p.plc_activation_rate * g_bg - 1.2 * plc
        dip3 = p.ip3_production_rate * plc - p.ip3_degradation_rate * ip3
        ddag = p.ip3_production_rate * plc - 0.4 * dag
        
        # calcium dynamics with hill function for cooperativity
        ca_release = p.ca_release_rate * (ip3**2 / (ip3**2 + 1.0))
        ca_reuptake = p.ca_reuptake_rate * ca
        dca = ca_release - ca_reuptake
        
        # mapk/erk pathway
        dmek = p.mek_activation_rate * g_alpha - 1.0 * mek
        derk = p.erk_activation_rate * mek - p.erk_deactivation_rate * erk
        
        # small gtpases (rho family)
        drac = p.rac_activation_rate * pip3 - p.rac_deactivation_rate * rac
        dcdc42 = p.cdc42_activation_rate * pip3 - p.cdc42_deactivation_rate * cdc42
        drho = 5.0 * g_alpha - 2.0 * rho
        
        return np.array([dg_alpha, dg_bg, dpi3k, dpip3, dakt,
                        dplc, dip3, ddag, dca, dmek, derk,
                        drac, dcdc42, drho])
    
    def integrate(self, g_protein_input: float, duration: float, 
                 dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        integrate signaling network over time
        
        args:
            g_protein_input: constant input signal
            duration: simulation time (s)
            dt: time step
        
        returns:
            time series of all state variables
        """
        # initial state
        y0 = np.array([
            self.state['g_alpha_gtp'],
            self.state['g_betagamma'],
            self.state['pi3k_active'],
            self.state['pip3'],
            self.state['akt_active'],
            self.state['plc_active'],
            self.state['ip3'],
            self.state['dag'],
            self.state['ca_cyto'],
            self.state['mek_active'],
            self.state['erk_active'],
            self.state['rac_gtp'],
            self.state['cdc42_gtp'],
            self.state['rho_gtp']
        ])
        
        # time points
        t = np.arange(0, duration, dt)
        
        # integrate
        solution = odeint(self.ode_system, y0, t, 
                         args=(g_protein_input,))
        
        # pack into dict
        keys = ['g_alpha_gtp', 'g_betagamma', 'pi3k_active', 'pip3',
                'akt_active', 'plc_active', 'ip3', 'dag', 'ca_cyto',
                'mek_active', 'erk_active', 'rac_gtp', 'cdc42_gtp', 'rho_gtp']
        
        result = {'time': t}
        for i, key in enumerate(keys):
            result[key] = solution[:, i]
        
        # update current state to final values
        for i, key in enumerate(keys):
            self.state[key] = solution[-1, i]
        
        return result
    
    def step(self, g_protein_input: float, dt: float):
        """
        single time step integration using euler method
        
        args:
            g_protein_input: current g-protein activation level
            dt: time step (s)
        """
        # current state vector
        y = np.array([
            self.state['g_alpha_gtp'],
            self.state['g_betagamma'],
            self.state['pi3k_active'],
            self.state['pip3'],
            self.state['akt_active'],
            self.state['plc_active'],
            self.state['ip3'],
            self.state['dag'],
            self.state['ca_cyto'],
            self.state['mek_active'],
            self.state['erk_active'],
            self.state['rac_gtp'],
            self.state['cdc42_gtp'],
            self.state['rho_gtp']
        ])
        
        # compute derivatives
        dydt = self.ode_system(y, 0, g_protein_input)
        
        # euler step
        y_new = y + dydt * dt
        
        # update state (ensure non-negative)
        keys = ['g_alpha_gtp', 'g_betagamma', 'pi3k_active', 'pip3',
                'akt_active', 'plc_active', 'ip3', 'dag', 'ca_cyto',
                'mek_active', 'erk_active', 'rac_gtp', 'cdc42_gtp', 'rho_gtp']
        
        for i, key in enumerate(keys):
            self.state[key] = max(0, y_new[i])
        
        # compute cellular outputs
        self._compute_outputs()
    
    def _compute_outputs(self):
        """
        compute functional outputs from pathway states
        """
        # migration: rac + cdc42 + pip3
        self.outputs['migration_signal'] = (
            0.4 * self.state['rac_gtp'] +
            0.3 * self.state['cdc42_gtp'] +
            0.3 * self.state['pip3']
        )
        
        # adhesion: akt + rho
        self.outputs['adhesion_signal'] = (
            0.6 * self.state['akt_active'] +
            0.4 * self.state['rho_gtp']
        )
        
        # transcription: erk
        self.outputs['transcription_signal'] = self.state['erk_active']
        
        # degranulation: calcium + dag
        self.outputs['degranulation_signal'] = (
            0.7 * self.state['ca_cyto'] +
            0.3 * self.state['dag']
        )
    
    def get_migration_bias(self) -> float:
        """
        returns normalized migration signal (0-1)
        """
        return np.tanh(self.outputs['migration_signal'] / 10.0)
    
    def get_adhesion_strength(self) -> float:
        """
        returns normalized adhesion signal (0-1)
        """
        return np.tanh(self.outputs['adhesion_signal'] / 8.0)


def ultrasensitive_response(input_signal: float, 
                           threshold: float = 1.0,
                           hill_coeff: float = 4.0) -> float:
    """
    ultrasensitive (switch-like) response using hill equation
    
    common in mapk cascades and other signaling modules
    
    args:
        input_signal: input concentration
        threshold: half-maximal activation
        hill_coeff: cooperativity (1=hyperbolic, >1=sigmoidal)
    
    returns:
        output signal (0-1)
    """
    return input_signal**hill_coeff / (threshold**hill_coeff + input_signal**hill_coeff)


def feedback_inhibition(signal: float, inhibitor: float,
                       ki: float = 1.0) -> float:
    """
    negative feedback inhibition
    
    args:
        signal: forward signal
        inhibitor: feedback inhibitor concentration
        ki: inhibition constant
    
    returns:
        inhibited signal
    """
    return signal / (1 + inhibitor / ki)


class AdaptationModule:
    """
    models perfect adaptation in chemotaxis via incoherent feedforward loop
    implements local excitation, global inhibition (legi) mechanism
    """
    
    def __init__(self, tau_fast: float = 2.0, tau_slow: float = 20.0):
        """
        args:
            tau_fast: fast excitation time constant (s)
            tau_slow: slow inhibition time constant (s)
        """
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        
        self.excitation = 0.0
        self.inhibition = 0.0
    
    def step(self, input_signal: float, dt: float) -> float:
        """
        compute adapted response
        
        returns:
            excitation - inhibition (can be negative)
        """
        # fast excitation
        self.excitation += (input_signal - self.excitation) * dt / self.tau_fast
        
        # slow inhibition (integrates excitation)
        self.inhibition += (self.excitation - self.inhibition) * dt / self.tau_slow
        
        # adapted output
        return self.excitation - self.inhibition
    
    def reset(self):
        """reset adaptation state"""
        self.excitation = 0.0
        self.inhibition = 0.0
