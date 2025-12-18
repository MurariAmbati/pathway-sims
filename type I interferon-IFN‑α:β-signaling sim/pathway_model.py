"""
Type I Interferon (IFN-α/β) Signaling Pathway Model

This module implements a detailed ODE-based model of Type I interferon signaling,
including receptor binding, JAK-STAT activation, ISG expression, and feedback loops.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PathwayParameters:
    """Parameters for the IFN-α/β signaling pathway model."""
    
    # IFN binding and receptor dynamics
    k_ifn_bind: float = 0.1          # IFN binding rate to IFNAR
    k_ifn_unbind: float = 0.01       # IFN unbinding rate
    k_ifnar_synth: float = 0.05      # IFNAR synthesis rate
    k_ifnar_deg: float = 0.02        # IFNAR degradation rate
    k_receptor_intern: float = 0.05  # Receptor internalization rate
    
    # JAK-STAT activation
    k_jak_phosph: float = 0.5        # JAK phosphorylation rate
    k_jak_dephosph: float = 0.1      # JAK dephosphorylation rate
    k_stat_phosph: float = 0.8       # STAT phosphorylation rate
    k_stat_dephosph: float = 0.15    # STAT dephosphorylation rate
    k_stat_dimer: float = 1.0        # STAT dimerization rate
    k_stat_dissoc: float = 0.05      # STAT dimer dissociation rate
    
    # ISGF3 complex formation
    k_isgf3_form: float = 0.7        # ISGF3 complex formation rate
    k_isgf3_dissoc: float = 0.08     # ISGF3 dissociation rate
    k_isgf3_nuclear: float = 0.9     # ISGF3 nuclear import rate
    k_isgf3_export: float = 0.05     # ISGF3 nuclear export rate
    
    # ISG transcription and expression
    k_isg_transcr: float = 1.2       # ISG transcription rate
    k_mrna_deg: float = 0.3          # mRNA degradation rate
    k_protein_synth: float = 0.8     # Protein synthesis rate
    k_protein_deg: float = 0.1       # Protein degradation rate
    
    # Negative feedback
    k_socs_synth: float = 0.6        # SOCS protein synthesis rate
    k_socs_inhib: float = 0.5        # SOCS inhibition strength
    k_ptp_activity: float = 0.2      # Protein tyrosine phosphatase activity
    
    # Positive feedback
    k_ifn_feedback: float = 0.3      # IFN positive feedback strength
    
    # Initial concentrations
    ifn_initial: float = 100.0       # Initial IFN concentration
    ifnar_initial: float = 1000.0    # Initial IFNAR concentration
    stat1_initial: float = 500.0     # Initial STAT1 concentration
    stat2_initial: float = 500.0     # Initial STAT2 concentration
    irf9_initial: float = 300.0      # Initial IRF9 concentration


class IFNSignalingModel:
    """
    Comprehensive model of Type I Interferon signaling pathway.
    
    Components modeled:
    - IFN-α/β binding to IFNAR1/IFNAR2 receptor complex
    - JAK1/TYK2 activation
    - STAT1/STAT2 phosphorylation and dimerization
    - ISGF3 complex formation (STAT1-STAT2-IRF9)
    - Nuclear translocation
    - ISG transcription and protein expression
    - Negative feedback via SOCS proteins
    - Positive feedback via IFN induction
    """
    
    def __init__(self, params: PathwayParameters = None):
        self.params = params or PathwayParameters()
        self.state_names = [
            'IFN', 'IFNAR', 'IFN_IFNAR', 'IFNAR_intern',
            'JAK_inactive', 'JAK_active',
            'STAT1', 'STAT2', 'pSTAT1', 'pSTAT2',
            'STAT1_STAT2', 'IRF9', 'ISGF3_cyto', 'ISGF3_nuc',
            'ISG_mRNA', 'ISG_protein',
            'SOCS', 'Antiviral_state'
        ]
    
    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Calculate derivatives for all species in the pathway.
        
        Args:
            state: Current state vector
            t: Current time
            
        Returns:
            Array of derivatives
        """
        # Unpack state variables
        (IFN, IFNAR, IFN_IFNAR, IFNAR_intern,
         JAK_inactive, JAK_active,
         STAT1, STAT2, pSTAT1, pSTAT2,
         STAT1_STAT2, IRF9, ISGF3_cyto, ISGF3_nuc,
         ISG_mRNA, ISG_protein,
         SOCS, Antiviral_state) = state
        
        p = self.params
        
        # IFN-IFNAR binding and receptor dynamics
        d_IFN = (-p.k_ifn_bind * IFN * IFNAR + 
                 p.k_ifn_unbind * IFN_IFNAR +
                 p.k_ifn_feedback * ISG_protein)
        
        d_IFNAR = (p.k_ifnar_synth - 
                   p.k_ifnar_deg * IFNAR -
                   p.k_ifn_bind * IFN * IFNAR +
                   p.k_ifn_unbind * IFN_IFNAR)
        
        d_IFN_IFNAR = (p.k_ifn_bind * IFN * IFNAR -
                       p.k_ifn_unbind * IFN_IFNAR -
                       p.k_receptor_intern * IFN_IFNAR)
        
        d_IFNAR_intern = p.k_receptor_intern * IFN_IFNAR - p.k_ifnar_deg * IFNAR_intern
        
        # JAK activation (inhibited by SOCS)
        socs_inhibition = 1.0 / (1.0 + p.k_socs_inhib * SOCS)
        
        d_JAK_inactive = (-p.k_jak_phosph * IFN_IFNAR * JAK_inactive * socs_inhibition +
                          p.k_jak_dephosph * JAK_active +
                          p.k_ptp_activity * JAK_active)
        
        d_JAK_active = (p.k_jak_phosph * IFN_IFNAR * JAK_inactive * socs_inhibition -
                        p.k_jak_dephosph * JAK_active -
                        p.k_ptp_activity * JAK_active)
        
        # STAT phosphorylation
        d_STAT1 = (-p.k_stat_phosph * JAK_active * STAT1 +
                   p.k_stat_dephosph * pSTAT1 +
                   p.k_stat_dissoc * STAT1_STAT2)
        
        d_STAT2 = (-p.k_stat_phosph * JAK_active * STAT2 +
                   p.k_stat_dephosph * pSTAT2 +
                   p.k_stat_dissoc * STAT1_STAT2)
        
        d_pSTAT1 = (p.k_stat_phosph * JAK_active * STAT1 -
                    p.k_stat_dephosph * pSTAT1 -
                    p.k_stat_dimer * pSTAT1 * pSTAT2)
        
        d_pSTAT2 = (p.k_stat_phosph * JAK_active * STAT2 -
                    p.k_stat_dephosph * pSTAT2 -
                    p.k_stat_dimer * pSTAT1 * pSTAT2)
        
        # STAT1-STAT2 heterodimer formation
        d_STAT1_STAT2 = (p.k_stat_dimer * pSTAT1 * pSTAT2 -
                         p.k_stat_dissoc * STAT1_STAT2 -
                         p.k_isgf3_form * STAT1_STAT2 * IRF9)
        
        # IRF9 dynamics
        d_IRF9 = (-p.k_isgf3_form * STAT1_STAT2 * IRF9 +
                  p.k_isgf3_dissoc * ISGF3_cyto +
                  p.k_isgf3_export * ISGF3_nuc)
        
        # ISGF3 complex formation and nuclear translocation
        d_ISGF3_cyto = (p.k_isgf3_form * STAT1_STAT2 * IRF9 -
                        p.k_isgf3_dissoc * ISGF3_cyto -
                        p.k_isgf3_nuclear * ISGF3_cyto +
                        p.k_isgf3_export * ISGF3_nuc)
        
        d_ISGF3_nuc = (p.k_isgf3_nuclear * ISGF3_cyto -
                       p.k_isgf3_export * ISGF3_nuc)
        
        # ISG transcription and translation
        d_ISG_mRNA = (p.k_isg_transcr * ISGF3_nuc -
                      p.k_mrna_deg * ISG_mRNA)
        
        d_ISG_protein = (p.k_protein_synth * ISG_mRNA -
                         p.k_protein_deg * ISG_protein)
        
        # SOCS negative feedback
        d_SOCS = (p.k_socs_synth * ISGF3_nuc -
                  p.k_protein_deg * SOCS)
        
        # Antiviral state (cumulative measure)
        d_Antiviral_state = 0.1 * ISG_protein
        
        return np.array([
            d_IFN, d_IFNAR, d_IFN_IFNAR, d_IFNAR_intern,
            d_JAK_inactive, d_JAK_active,
            d_STAT1, d_STAT2, d_pSTAT1, d_pSTAT2,
            d_STAT1_STAT2, d_IRF9, d_ISGF3_cyto, d_ISGF3_nuc,
            d_ISG_mRNA, d_ISG_protein,
            d_SOCS, d_Antiviral_state
        ])
    
    def get_initial_state(self) -> np.ndarray:
        """Get initial state vector."""
        p = self.params
        return np.array([
            p.ifn_initial,      # IFN
            p.ifnar_initial,    # IFNAR
            0.0,                # IFN_IFNAR
            0.0,                # IFNAR_intern
            1000.0,             # JAK_inactive
            0.0,                # JAK_active
            p.stat1_initial,    # STAT1
            p.stat2_initial,    # STAT2
            0.0,                # pSTAT1
            0.0,                # pSTAT2
            0.0,                # STAT1_STAT2
            p.irf9_initial,     # IRF9
            0.0,                # ISGF3_cyto
            0.0,                # ISGF3_nuc
            0.0,                # ISG_mRNA
            0.0,                # ISG_protein
            0.0,                # SOCS
            0.0                 # Antiviral_state
        ])
    
    def simulate(self, t_span: Tuple[float, float], n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation of the pathway.
        
        Args:
            t_span: Tuple of (start_time, end_time)
            n_points: Number of time points
            
        Returns:
            Tuple of (time_array, state_array)
        """
        t = np.linspace(t_span[0], t_span[1], n_points)
        y0 = self.get_initial_state()
        
        solution = odeint(self.derivatives, y0, t)
        
        return t, solution
    
    def get_state_dict(self, solution: np.ndarray, time_idx: int = -1) -> Dict[str, float]:
        """
        Convert state array to dictionary.
        
        Args:
            solution: Solution array from simulation
            time_idx: Time index to extract (default: last time point)
            
        Returns:
            Dictionary mapping state names to values
        """
        return {name: solution[time_idx, i] for i, name in enumerate(self.state_names)}


def analyze_pathway_dynamics(model: IFNSignalingModel, 
                             solution: np.ndarray,
                             t: np.ndarray) -> Dict[str, any]:
    """
    Analyze key features of the pathway dynamics.
    
    Args:
        model: IFNSignalingModel instance
        solution: Solution array from simulation
        t: Time array
        
    Returns:
        Dictionary containing analysis results
    """
    # Extract key components
    isgf3_nuc = solution[:, model.state_names.index('ISGF3_nuc')]
    isg_protein = solution[:, model.state_names.index('ISG_protein')]
    jak_active = solution[:, model.state_names.index('JAK_active')]
    socs = solution[:, model.state_names.index('SOCS')]
    
    # Find peaks
    isgf3_peak_idx = np.argmax(isgf3_nuc)
    isg_peak_idx = np.argmax(isg_protein)
    
    # Calculate response times
    isgf3_threshold = 0.1 * np.max(isgf3_nuc)
    isg_threshold = 0.1 * np.max(isg_protein)
    
    isgf3_response_time = t[np.where(isgf3_nuc > isgf3_threshold)[0][0]] if np.any(isgf3_nuc > isgf3_threshold) else None
    isg_response_time = t[np.where(isg_protein > isg_threshold)[0][0]] if np.any(isg_protein > isg_threshold) else None
    
    return {
        'isgf3_peak_value': np.max(isgf3_nuc),
        'isgf3_peak_time': t[isgf3_peak_idx],
        'isg_peak_value': np.max(isg_protein),
        'isg_peak_time': t[isg_peak_idx],
        'isgf3_response_time': isgf3_response_time,
        'isg_response_time': isg_response_time,
        'jak_max_activation': np.max(jak_active),
        'socs_feedback_strength': np.max(socs),
        'steady_state_isg': isg_protein[-1],
    }
