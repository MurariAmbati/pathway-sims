"""
FcγR Signaling Models
Computational models for Fc receptor signaling pathways in innate immune cells
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class FcgRParameters:
    """Parameters for FcγR signaling cascade"""
    # Receptor parameters
    fcgr_density: float = 50000  # receptors per cell
    fcgr_affinity: float = 1e-7  # M (Kd for IgG binding)
    
    # ITAM signaling
    itam_phosphorylation_rate: float = 0.1  # s^-1
    itam_dephosphorylation_rate: float = 0.05  # s^-1
    
    # Syk kinase activation
    syk_activation_rate: float = 0.5  # s^-1
    syk_deactivation_rate: float = 0.1  # s^-1
    syk_total: float = 10000  # molecules per cell
    
    # PI3K/Akt pathway
    pi3k_activation_rate: float = 0.3  # s^-1
    pi3k_deactivation_rate: float = 0.15  # s^-1
    akt_activation_rate: float = 0.4  # s^-1
    akt_deactivation_rate: float = 0.2  # s^-1
    
    # MAPK cascade
    mek_activation_rate: float = 0.35  # s^-1
    mek_deactivation_rate: float = 0.18  # s^-1
    erk_activation_rate: float = 0.45  # s^-1
    erk_deactivation_rate: float = 0.22  # s^-1
    
    # Calcium signaling
    plcg_activation_rate: float = 0.25  # s^-1
    ip3_production_rate: float = 1.0  # s^-1
    ip3_degradation_rate: float = 0.5  # s^-1
    ca_release_rate: float = 2.0  # s^-1
    ca_reuptake_rate: float = 1.5  # s^-1
    ca_basal: float = 0.1  # μM
    
    # Transcription factors
    nfat_activation_rate: float = 0.2  # s^-1
    nfat_deactivation_rate: float = 0.1  # s^-1
    nfkb_activation_rate: float = 0.3  # s^-1
    nfkb_deactivation_rate: float = 0.15  # s^-1
    
    # Cytokine production
    tnf_production_rate: float = 0.1  # s^-1
    il6_production_rate: float = 0.08  # s^-1
    il1b_production_rate: float = 0.12  # s^-1
    
    # Inhibitory receptor parameters
    fcgr2b_ratio: float = 0.2  # ratio to activating FcγR
    ship_activation_rate: float = 0.4  # s^-1


class FcgRSignalingModel:
    """
    Comprehensive model of FcγR signaling cascade in innate immune cells
    Includes ITAM phosphorylation, Syk activation, and downstream pathways
    """
    
    def __init__(self, params: Optional[FcgRParameters] = None):
        self.params = params or FcgRParameters()
        
    def receptor_binding(self, ic_concentration: float, time_points: np.ndarray) -> np.ndarray:
        """
        Calculate FcγR-immune complex binding kinetics
        
        Args:
            ic_concentration: Immune complex concentration (M)
            time_points: Time points for simulation (s)
            
        Returns:
            Bound receptor fraction over time
        """
        kon = 1e5  # M^-1 s^-1
        koff = kon * self.params.fcgr_affinity
        
        def binding_ode(bound, t):
            free_receptors = 1.0 - bound
            return kon * ic_concentration * free_receptors - koff * bound
        
        bound_fraction = odeint(binding_ode, 0.0, time_points)
        return bound_fraction.flatten()
    
    def signaling_cascade(self, 
                         ic_concentration: float,
                         time_points: np.ndarray,
                         include_inhibitory: bool = False) -> Dict[str, np.ndarray]:
        """
        Simulate complete FcγR signaling cascade with ODEs
        
        State variables:
        0: Bound receptors
        1: Phosphorylated ITAM
        2: Active Syk
        3: Active PI3K
        4: Active Akt
        5: Active MEK
        6: Active ERK
        7: Active PLCγ
        8: IP3
        9: Cytosolic Ca2+
        10: Active NFAT
        11: Active NF-κB
        12: TNF-α
        13: IL-6
        14: IL-1β
        15: SHIP (inhibitory)
        """
        
        def odes(y, t):
            bound, p_itam, syk_a, pi3k_a, akt_a, mek_a, erk_a, plcg_a, ip3, ca, nfat_a, nfkb_a, tnf, il6, il1b, ship_a = y
            
            # Receptor binding
            free_receptors = 1.0 - bound
            kon = 1e5  # M^-1 s^-1
            koff = kon * self.params.fcgr_affinity
            d_bound = kon * ic_concentration * free_receptors - koff * bound
            
            # ITAM phosphorylation (proportional to bound receptors)
            d_p_itam = (self.params.itam_phosphorylation_rate * bound * (1 - p_itam) - 
                       self.params.itam_dephosphorylation_rate * p_itam)
            
            # Syk activation (recruited by phosphorylated ITAM)
            syk_inactive = 1.0 - syk_a
            inhibition = 1.0
            if include_inhibitory:
                inhibition = 1.0 / (1.0 + 5.0 * ship_a)  # SHIP inhibits Syk pathway
            
            d_syk_a = (self.params.syk_activation_rate * p_itam * syk_inactive * inhibition - 
                      self.params.syk_deactivation_rate * syk_a)
            
            # PI3K/Akt pathway
            pi3k_inactive = 1.0 - pi3k_a
            d_pi3k_a = (self.params.pi3k_activation_rate * syk_a * pi3k_inactive - 
                       self.params.pi3k_deactivation_rate * pi3k_a)
            
            akt_inactive = 1.0 - akt_a
            d_akt_a = (self.params.akt_activation_rate * pi3k_a * akt_inactive - 
                      self.params.akt_deactivation_rate * akt_a)
            
            # MAPK cascade
            mek_inactive = 1.0 - mek_a
            d_mek_a = (self.params.mek_activation_rate * syk_a * mek_inactive - 
                      self.params.mek_deactivation_rate * mek_a)
            
            erk_inactive = 1.0 - erk_a
            d_erk_a = (self.params.erk_activation_rate * mek_a * erk_inactive - 
                      self.params.erk_deactivation_rate * erk_a)
            
            # Calcium signaling
            plcg_inactive = 1.0 - plcg_a
            d_plcg_a = (self.params.plcg_activation_rate * syk_a * plcg_inactive - 
                       self.params.plcg_activation_rate * 0.5 * plcg_a)
            
            d_ip3 = (self.params.ip3_production_rate * plcg_a - 
                    self.params.ip3_degradation_rate * ip3)
            
            d_ca = (self.params.ca_release_rate * ip3 * (10.0 - ca) - 
                   self.params.ca_reuptake_rate * (ca - self.params.ca_basal))
            
            # Transcription factors
            nfat_inactive = 1.0 - nfat_a
            ca_factor = ca / (ca + 0.5)  # Ca2+ dependent activation
            d_nfat_a = (self.params.nfat_activation_rate * ca_factor * nfat_inactive - 
                       self.params.nfat_deactivation_rate * nfat_a)
            
            nfkb_inactive = 1.0 - nfkb_a
            erk_factor = erk_a / (erk_a + 0.3)
            d_nfkb_a = (self.params.nfkb_activation_rate * erk_factor * nfkb_inactive - 
                       self.params.nfkb_deactivation_rate * nfkb_a)
            
            # Cytokine production
            d_tnf = self.params.tnf_production_rate * nfkb_a * (1.0 + nfat_a)
            d_il6 = self.params.il6_production_rate * nfkb_a * (1.0 + akt_a)
            d_il1b = self.params.il1b_production_rate * nfkb_a
            
            # Inhibitory SHIP activation (from FcγRIIB)
            if include_inhibitory:
                ship_inactive = 1.0 - ship_a
                d_ship_a = (self.params.ship_activation_rate * bound * self.params.fcgr2b_ratio * ship_inactive - 
                           0.2 * ship_a)
            else:
                d_ship_a = 0.0
            
            return [d_bound, d_p_itam, d_syk_a, d_pi3k_a, d_akt_a, d_mek_a, d_erk_a, 
                   d_plcg_a, d_ip3, d_ca, d_nfat_a, d_nfkb_a, d_tnf, d_il6, d_il1b, d_ship_a]
        
        # Initial conditions (all inactive/zero except basal calcium)
        y0 = [0.0] * 16
        y0[9] = self.params.ca_basal
        
        # Solve ODEs
        solution = odeint(odes, y0, time_points)
        
        return {
            'bound_receptors': solution[:, 0],
            'phospho_itam': solution[:, 1],
            'active_syk': solution[:, 2],
            'active_pi3k': solution[:, 3],
            'active_akt': solution[:, 4],
            'active_mek': solution[:, 5],
            'active_erk': solution[:, 6],
            'active_plcg': solution[:, 7],
            'ip3': solution[:, 8],
            'calcium': solution[:, 9],
            'active_nfat': solution[:, 10],
            'active_nfkb': solution[:, 11],
            'tnf_alpha': solution[:, 12],
            'il6': solution[:, 13],
            'il1_beta': solution[:, 14],
            'active_ship': solution[:, 15]
        }


class AntibodyDependentPhagocytosis:
    """
    Model for antibody-dependent cellular phagocytosis (ADCP)
    """
    
    def __init__(self):
        self.params = FcgRParameters()
        
    def phagocytosis_efficiency(self,
                               antibody_concentration: np.ndarray,
                               antigen_density: float,
                               fcgr_density: float,
                               cell_type: str = 'macrophage') -> np.ndarray:
        """
        Calculate phagocytosis efficiency based on antibody opsonization
        
        Args:
            antibody_concentration: IgG concentrations (μg/mL)
            antigen_density: Target antigens per target cell
            fcgr_density: FcγR per effector cell
            cell_type: Type of phagocyte (macrophage, neutrophil, monocyte)
            
        Returns:
            Phagocytosis efficiency (0-1)
        """
        # Cell type specific parameters
        type_params = {
            'macrophage': {'fcgr1': 0.3, 'fcgr2a': 0.5, 'fcgr2b': 0.1, 'fcgr3a': 0.1},
            'neutrophil': {'fcgr1': 0.2, 'fcgr2a': 0.4, 'fcgr2b': 0.05, 'fcgr3b': 0.35},
            'monocyte': {'fcgr1': 0.25, 'fcgr2a': 0.45, 'fcgr2b': 0.15, 'fcgr3a': 0.15}
        }
        
        ratios = type_params.get(cell_type, type_params['macrophage'])
        
        # Calculate opsonization level (antibodies bound to target)
        ka = 1e8  # M^-1 (affinity constant)
        mw_igg = 150000  # g/mol
        ab_molar = (antibody_concentration * 1e-6) / mw_igg  # M
        
        opsonization = (ka * ab_molar * antigen_density) / (1 + ka * ab_molar)
        
        # Calculate activating signal
        activating = (ratios.get('fcgr1', 0) * 1.0 + 
                     ratios.get('fcgr2a', 0) * 0.8 + 
                     ratios.get('fcgr3a', 0) * 0.6 + 
                     ratios.get('fcgr3b', 0) * 0.4)
        
        # Calculate inhibitory signal
        inhibitory = ratios.get('fcgr2b', 0) * 0.5
        
        # Net signal
        net_signal = activating - inhibitory
        
        # Phagocytosis efficiency with Hill equation
        efficiency = (opsonization * net_signal) / (0.5 + opsonization * net_signal)
        
        return efficiency
    
    def phagocytosis_kinetics(self,
                             opsonization_level: float,
                             time_points: np.ndarray,
                             effector_target_ratio: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Model phagocytosis kinetics over time
        
        Args:
            opsonization_level: Antibody opsonization level (0-1)
            time_points: Time points (minutes)
            effector_target_ratio: Ratio of effector to target cells
            
        Returns:
            Dictionary with phagocytosis kinetics
        """
        # Rate constants
        k_bind = 0.1 * opsonization_level  # min^-1
        k_engulf = 0.05 * opsonization_level  # min^-1
        k_digest = 0.02  # min^-1
        
        def kinetics_ode(y, t):
            free_targets, bound_targets, engulfed_targets, digested_targets = y
            
            available_effectors = effector_target_ratio - bound_targets - engulfed_targets
            available_effectors = max(0, available_effectors)
            
            # Binding
            d_free = -k_bind * free_targets * available_effectors
            
            # Engulfment
            d_bound = (k_bind * free_targets * available_effectors - 
                      k_engulf * bound_targets)
            
            # Digestion
            d_engulf = k_engulf * bound_targets - k_digest * engulfed_targets
            
            # Completed
            d_digest = k_digest * engulfed_targets
            
            return [d_free, d_bound, d_engulf, d_digest]
        
        # Initial conditions: all targets are free
        y0 = [1.0, 0.0, 0.0, 0.0]
        
        solution = odeint(kinetics_ode, y0, time_points)
        
        return {
            'free_targets': solution[:, 0],
            'bound_targets': solution[:, 1],
            'engulfed_targets': solution[:, 2],
            'digested_targets': solution[:, 3],
            'total_phagocytosed': solution[:, 2] + solution[:, 3]
        }


class AntibodyDependentCytotoxicity:
    """
    Model for antibody-dependent cellular cytotoxicity (ADCC)
    """
    
    def __init__(self):
        self.params = FcgRParameters()
        
    def adcc_efficiency(self,
                       antibody_concentration: np.ndarray,
                       target_density: float,
                       effector_type: str = 'nk_cell') -> np.ndarray:
        """
        Calculate ADCC efficiency
        
        Args:
            antibody_concentration: IgG concentrations (μg/mL)
            target_density: Target antigens per cell
            effector_type: Type of effector cell (nk_cell, macrophage, neutrophil)
            
        Returns:
            ADCC efficiency (0-1)
        """
        # Effector type specific parameters
        type_params = {
            'nk_cell': {'fcgr3a': 1.0, 'affinity_factor': 1.0, 'cytotoxic_potential': 1.0},
            'macrophage': {'fcgr1': 0.4, 'fcgr2a': 0.4, 'fcgr3a': 0.2, 
                          'affinity_factor': 0.7, 'cytotoxic_potential': 0.6},
            'neutrophil': {'fcgr2a': 0.5, 'fcgr3b': 0.5,
                          'affinity_factor': 0.6, 'cytotoxic_potential': 0.5}
        }
        
        params = type_params.get(effector_type, type_params['nk_cell'])
        
        # Calculate opsonization
        ka = 1e8  # M^-1
        mw_igg = 150000
        ab_molar = (antibody_concentration * 1e-6) / mw_igg
        
        opsonization = (ka * ab_molar * target_density) / (1 + ka * ab_molar)
        
        # Calculate cytotoxicity with cooperative binding
        hill_coefficient = 2.0  # Cooperative FcγR engagement
        ec50 = 0.3
        
        efficiency = (params['cytotoxic_potential'] * 
                     (opsonization ** hill_coefficient) / 
                     (ec50 ** hill_coefficient + opsonization ** hill_coefficient))
        
        return efficiency
    
    def cytotoxicity_kinetics(self,
                             opsonization_level: float,
                             time_points: np.ndarray,
                             effector_target_ratio: float = 10.0,
                             effector_type: str = 'nk_cell') -> Dict[str, np.ndarray]:
        """
        Model ADCC kinetics over time
        
        Args:
            opsonization_level: Antibody opsonization level (0-1)
            time_points: Time points (hours)
            effector_target_ratio: Ratio of effector to target cells
            effector_type: Type of effector cell
            
        Returns:
            Dictionary with cytotoxicity kinetics
        """
        # Effector-specific parameters
        if effector_type == 'nk_cell':
            k_bind = 0.5 * opsonization_level  # h^-1
            k_kill = 0.3  # h^-1
            k_detach = 0.2  # h^-1
        elif effector_type == 'macrophage':
            k_bind = 0.3 * opsonization_level
            k_kill = 0.15
            k_detach = 0.25
        else:  # neutrophil
            k_bind = 0.4 * opsonization_level
            k_kill = 0.2
            k_detach = 0.3
        
        def cytotoxicity_ode(y, t):
            viable_targets, conjugated, dead_targets, exhausted_effectors = y
            
            # Available effectors
            available_effectors = effector_target_ratio - conjugated - exhausted_effectors
            available_effectors = max(0, available_effectors)
            
            # Conjugate formation
            d_viable = -k_bind * viable_targets * available_effectors
            
            # Killing and detachment
            d_conjugated = (k_bind * viable_targets * available_effectors - 
                           (k_kill + k_detach) * conjugated)
            
            # Target death
            d_dead = k_kill * conjugated
            
            # Effector exhaustion (some effectors become exhausted after killing)
            d_exhausted = 0.3 * k_kill * conjugated
            
            return [d_viable, d_conjugated, d_dead, d_exhausted]
        
        # Initial conditions
        y0 = [1.0, 0.0, 0.0, 0.0]
        
        solution = odeint(cytotoxicity_ode, y0, time_points)
        
        # Calculate specific lysis percentage
        specific_lysis = 100 * solution[:, 2]
        
        return {
            'viable_targets': solution[:, 0],
            'conjugated': solution[:, 1],
            'dead_targets': solution[:, 2],
            'exhausted_effectors': solution[:, 3],
            'specific_lysis': specific_lysis
        }


class FcgRCrosslinking:
    """
    Model for FcγR crosslinking and clustering dynamics
    """
    
    def __init__(self):
        self.params = FcgRParameters()
        
    def clustering_dynamics(self,
                           ic_density: float,
                           time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate FcγR clustering upon immune complex binding
        
        Args:
            ic_density: Immune complex density (arbitrary units)
            time_points: Time points (seconds)
            
        Returns:
            Clustering dynamics
        """
        # Rate constants for cluster formation
        k_on = 0.1 * ic_density  # Cluster formation rate
        k_off = 0.05  # Cluster dissociation rate
        k_growth = 0.2  # Cluster growth rate
        k_fusion = 0.05  # Cluster fusion rate
        
        def cluster_ode(y, t):
            monomers, dimers, trimers, large_clusters = y
            
            # Monomer-monomer -> dimer
            d_monomers = (-2 * k_on * monomers**2 + 
                         2 * k_off * dimers - 
                         k_on * monomers * dimers)
            
            # Dimer formation and growth
            d_dimers = (k_on * monomers**2 - 
                       k_off * dimers - 
                       k_on * monomers * dimers - 
                       k_growth * dimers + 
                       k_off * trimers)
            
            # Trimer formation
            d_trimers = (k_on * monomers * dimers + 
                        k_growth * dimers - 
                        k_off * trimers - 
                        k_growth * trimers)
            
            # Large clusters (>3 receptors)
            d_large = (k_growth * trimers + 
                      k_fusion * trimers**2)
            
            return [d_monomers, d_dimers, d_trimers, d_large]
        
        # Initial conditions: all receptors as monomers
        y0 = [1.0, 0.0, 0.0, 0.0]
        
        solution = odeint(cluster_ode, y0, time_points)
        
        # Calculate average cluster size
        avg_size = (1 * solution[:, 0] + 
                   2 * solution[:, 1] + 
                   3 * solution[:, 2] + 
                   5 * solution[:, 3])
        
        return {
            'monomers': solution[:, 0],
            'dimers': solution[:, 1],
            'trimers': solution[:, 2],
            'large_clusters': solution[:, 3],
            'average_cluster_size': avg_size
        }
