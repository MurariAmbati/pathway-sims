"""
Progesterone Receptor Signaling Model
Implements receptor dynamics, ligand binding, and downstream signaling cascades
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class ReceptorParameters:
    """Parameters for progesterone receptor signaling"""
    # Receptor isoforms (PR-A and PR-B)
    PR_A_total: float = 1000.0  # nM
    PR_B_total: float = 1500.0  # nM
    
    # Binding kinetics
    k_on: float = 0.1  # nM^-1 min^-1
    k_off: float = 0.01  # min^-1
    K_d: float = 0.1  # nM (dissociation constant)
    
    # Dimerization
    k_dimer: float = 0.05  # nM^-1 min^-1
    k_dimer_dissoc: float = 0.001  # min^-1
    
    # Nuclear translocation
    k_import: float = 0.02  # min^-1
    k_export: float = 0.005  # min^-1
    
    # DNA binding and transcription
    k_dna_bind: float = 0.1  # nM^-1 min^-1
    k_dna_dissoc: float = 0.01  # min^-1
    k_transcription: float = 0.5  # min^-1
    
    # mRNA and protein dynamics
    k_mRNA_deg: float = 0.05  # min^-1
    k_translation: float = 0.1  # min^-1
    k_protein_deg: float = 0.01  # min^-1
    
    # Cofactor recruitment
    k_coact_bind: float = 0.08  # nM^-1 min^-1
    k_coact_dissoc: float = 0.02  # min^-1
    coactivator_total: float = 500.0  # nM
    
    # Phosphorylation (MAPK, CDK2 pathways)
    k_phosph: float = 0.03  # min^-1
    k_dephosph: float = 0.01  # min^-1


class ProgesteroneReceptorModel:
    """Comprehensive progesterone receptor signaling model"""
    
    def __init__(self, params: ReceptorParameters = None):
        self.params = params or ReceptorParameters()
        
    def receptor_dynamics(self, state: np.ndarray, t: float, 
                         progesterone: float) -> np.ndarray:
        """
        ODE system for progesterone receptor signaling
        
        State variables:
        0: PR-A cytoplasmic (unbound)
        1: PR-B cytoplasmic (unbound)
        2: PR-A:P4 complex (cytoplasmic)
        3: PR-B:P4 complex (cytoplasmic)
        4: PR-A:P4 dimer (cytoplasmic)
        5: PR-B:P4 dimer (cytoplasmic)
        6: PR-A:P4 dimer (nuclear)
        7: PR-B:P4 dimer (nuclear)
        8: PR:DNA complex
        9: PR:DNA:Coactivator complex
        10: Target mRNA
        11: Target protein
        12: Phosphorylated PR-A
        13: Phosphorylated PR-B
        """
        p = self.params
        
        PR_A_cyto = state[0]
        PR_B_cyto = state[1]
        PR_A_P4 = state[2]
        PR_B_P4 = state[3]
        PR_A_dimer = state[4]
        PR_B_dimer = state[5]
        PR_A_nuc = state[6]
        PR_B_nuc = state[7]
        PR_DNA = state[8]
        PR_DNA_coact = state[9]
        mRNA = state[10]
        protein = state[11]
        PR_A_P = state[12]
        PR_B_P = state[13]
        
        dstate = np.zeros(14)
        
        # Ligand binding
        dstate[0] = -p.k_on * PR_A_cyto * progesterone + p.k_off * PR_A_P4 - p.k_phosph * PR_A_cyto + p.k_dephosph * PR_A_P
        dstate[1] = -p.k_on * PR_B_cyto * progesterone + p.k_off * PR_B_P4 - p.k_phosph * PR_B_cyto + p.k_dephosph * PR_B_P
        dstate[2] = p.k_on * PR_A_cyto * progesterone - p.k_off * PR_A_P4 - 2 * p.k_dimer * PR_A_P4 * PR_A_P4 + 2 * p.k_dimer_dissoc * PR_A_dimer
        dstate[3] = p.k_on * PR_B_cyto * progesterone - p.k_off * PR_B_P4 - 2 * p.k_dimer * PR_B_P4 * PR_B_P4 + 2 * p.k_dimer_dissoc * PR_B_dimer
        
        # Dimerization
        dstate[4] = p.k_dimer * PR_A_P4 * PR_A_P4 - p.k_dimer_dissoc * PR_A_dimer - p.k_import * PR_A_dimer + p.k_export * PR_A_nuc
        dstate[5] = p.k_dimer * PR_B_P4 * PR_B_P4 - p.k_dimer_dissoc * PR_B_dimer - p.k_import * PR_B_dimer + p.k_export * PR_B_nuc
        
        # Nuclear translocation
        dstate[6] = p.k_import * PR_A_dimer - p.k_export * PR_A_nuc - p.k_dna_bind * PR_A_nuc + p.k_dna_dissoc * PR_DNA * 0.4
        dstate[7] = p.k_import * PR_B_dimer - p.k_export * PR_B_nuc - p.k_dna_bind * PR_B_nuc + p.k_dna_dissoc * PR_DNA * 0.6
        
        # DNA binding (PR-B has higher transcriptional activity)
        dstate[8] = p.k_dna_bind * (PR_A_nuc * 0.4 + PR_B_nuc * 0.6) - p.k_dna_dissoc * PR_DNA - p.k_coact_bind * PR_DNA * (p.coactivator_total - PR_DNA_coact) + p.k_coact_dissoc * PR_DNA_coact
        
        # Coactivator recruitment
        dstate[9] = p.k_coact_bind * PR_DNA * (p.coactivator_total - PR_DNA_coact) - p.k_coact_dissoc * PR_DNA_coact
        
        # Transcription and translation
        dstate[10] = p.k_transcription * PR_DNA_coact - p.k_mRNA_deg * mRNA
        dstate[11] = p.k_translation * mRNA - p.k_protein_deg * protein
        
        # Phosphorylation
        dstate[12] = p.k_phosph * PR_A_cyto - p.k_dephosph * PR_A_P
        dstate[13] = p.k_phosph * PR_B_cyto - p.k_dephosph * PR_B_P
        
        return dstate
    
    def simulate(self, progesterone_concentration: float, 
                 time_points: np.ndarray,
                 initial_state: np.ndarray = None) -> np.ndarray:
        """Simulate the receptor signaling cascade"""
        
        if initial_state is None:
            # Initial conditions: all receptors in cytoplasm, unbound
            initial_state = np.array([
                self.params.PR_A_total,  # PR-A cytoplasmic
                self.params.PR_B_total,  # PR-B cytoplasmic
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Rest at zero
            ])
        
        solution = odeint(
            self.receptor_dynamics,
            initial_state,
            time_points,
            args=(progesterone_concentration,)
        )
        
        return solution
    
    def calculate_metrics(self, solution: np.ndarray) -> Dict[str, float]:
        """Calculate key signaling metrics"""
        
        # Peak nuclear receptor
        peak_nuclear = np.max(solution[:, 6] + solution[:, 7])
        
        # Time to half-maximal response
        target_protein = solution[:, 11]
        max_protein = np.max(target_protein)
        if max_protein > 0:
            half_max_idx = np.where(target_protein >= max_protein / 2)[0]
            t_half = half_max_idx[0] if len(half_max_idx) > 0 else 0
        else:
            t_half = 0
        
        # Steady-state protein level
        ss_protein = np.mean(target_protein[-10:])
        
        # PR-A to PR-B ratio in nucleus
        PR_A_nuc_final = solution[-1, 6]
        PR_B_nuc_final = solution[-1, 7]
        ratio = PR_A_nuc_final / PR_B_nuc_final if PR_B_nuc_final > 0 else 0
        
        return {
            'peak_nuclear_receptor': peak_nuclear,
            'time_to_half_max': t_half,
            'steady_state_protein': ss_protein,
            'PR_A_to_PR_B_ratio': ratio,
            'total_dna_bound': np.max(solution[:, 8])
        }


class TissueSpecificModel:
    """Tissue-specific progesterone receptor signaling"""
    
    @staticmethod
    def uterine_model() -> ProgesteroneReceptorModel:
        """Uterine tissue - high PR expression during pregnancy"""
        params = ReceptorParameters()
        params.PR_A_total = 2000.0  # Higher expression
        params.PR_B_total = 3000.0  # PR-B dominant in uterus
        params.k_transcription = 0.7  # Higher transcriptional activity
        return ProgesteroneReceptorModel(params)
    
    @staticmethod
    def breast_model() -> ProgesteroneReceptorModel:
        """Breast tissue - balanced PR isoforms"""
        params = ReceptorParameters()
        params.PR_A_total = 1500.0
        params.PR_B_total = 1500.0  # Balanced expression
        params.k_transcription = 0.4
        return ProgesteroneReceptorModel(params)
    
    @staticmethod
    def pregnancy_model(trimester: int = 1) -> ProgesteroneReceptorModel:
        """Pregnancy-specific signaling (varies by trimester)"""
        params = ReceptorParameters()
        
        if trimester == 1:
            params.PR_A_total = 1500.0
            params.PR_B_total = 2500.0
        elif trimester == 2:
            params.PR_A_total = 3000.0
            params.PR_B_total = 4000.0
        else:  # trimester 3
            params.PR_A_total = 4000.0
            params.PR_B_total = 5000.0
            params.k_transcription = 0.8
        
        return ProgesteroneReceptorModel(params)
