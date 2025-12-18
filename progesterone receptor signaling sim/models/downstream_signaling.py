"""
Downstream signaling pathways and gene regulation
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class GeneTargets:
    """Progesterone-regulated gene targets"""
    
    # Pregnancy and uterine function genes
    PREGNANCY_GENES = [
        'HAND2',  # Decidualization
        'HOXA10', # Implantation
        'LIF',    # Leukemia inhibitory factor
        'PROKR1', # Prokineticin receptor 1
        'PTGS2',  # Prostaglandin synthesis
        'VEGFA',  # Angiogenesis
    ]
    
    # Breast development genes
    BREAST_GENES = [
        'RANKL',  # Mammary gland development
        'WNT4',   # Ductal branching
        'AREG',   # Amphiregulin
        'CCND1',  # Cyclin D1
        'MYC',    # Cell proliferation
    ]
    
    # Cell cycle and proliferation
    PROLIFERATION_GENES = [
        'CCNA1', 'CCNB1', 'CDK2', 'CDK4', 'E2F1'
    ]
    
    # Anti-inflammatory response
    IMMUNE_GENES = [
        'IL10', 'TGFB1', 'FOXP3', 'IL1RN'
    ]


class DownstreamSignaling:
    """Models downstream signaling cascades"""
    
    def __init__(self):
        self.mapk_activation = 0.0
        self.pi3k_activation = 0.0
        self.src_activation = 0.0
        
    def rapid_signaling(self, PR_membrane: float, time: float) -> Dict[str, float]:
        """
        Non-genomic (rapid) signaling through membrane PR
        Activates MAPK, PI3K/AKT, and Src pathways
        """
        
        # MAPK/ERK pathway
        k_mapk = 0.05
        mapk = PR_membrane * k_mapk * (1 - np.exp(-0.1 * time))
        
        # PI3K/AKT pathway
        k_pi3k = 0.03
        pi3k = PR_membrane * k_pi3k * (1 - np.exp(-0.08 * time))
        
        # Src kinase pathway
        k_src = 0.04
        src = PR_membrane * k_src * (1 - np.exp(-0.12 * time))
        
        return {
            'MAPK': mapk,
            'PI3K_AKT': pi3k,
            'Src': src
        }
    
    def gene_expression_profile(self, PR_DNA_bound: float, 
                                tissue_type: str = 'uterine') -> Dict[str, float]:
        """
        Calculate gene expression profiles based on PR-DNA binding
        """
        
        genes = GeneTargets()
        expression = {}
        
        if tissue_type == 'uterine':
            for gene in genes.PREGNANCY_GENES:
                # Different genes have different sensitivities
                sensitivity = np.random.uniform(0.5, 1.5)
                expression[gene] = PR_DNA_bound * sensitivity
                
        elif tissue_type == 'breast':
            for gene in genes.BREAST_GENES:
                sensitivity = np.random.uniform(0.5, 1.5)
                expression[gene] = PR_DNA_bound * sensitivity
        
        # Common proliferation response
        for gene in genes.PROLIFERATION_GENES:
            sensitivity = np.random.uniform(0.3, 1.0)
            expression[gene] = PR_DNA_bound * sensitivity
        
        return expression
    
    def calculate_proliferation_index(self, gene_expression: Dict[str, float]) -> float:
        """Calculate cell proliferation index from gene expression"""
        
        proliferation_genes = ['CCNA1', 'CCNB1', 'CDK2', 'CDK4', 'E2F1', 'MYC']
        total = sum(gene_expression.get(g, 0) for g in proliferation_genes)
        
        # Normalize to 0-100 scale
        return min(100, total / 10)


class CellularResponse:
    """Models cellular responses to progesterone signaling"""
    
    @staticmethod
    def decidualization_score(gene_expression: Dict[str, float]) -> float:
        """
        Calculate decidualization score for uterine stromal cells
        Based on HAND2, HOXA10, and PROKR1 expression
        """
        key_genes = ['HAND2', 'HOXA10', 'PROKR1']
        score = sum(gene_expression.get(g, 0) for g in key_genes) / len(key_genes)
        return min(100, score * 10)
    
    @staticmethod
    def implantation_receptivity(gene_expression: Dict[str, float]) -> float:
        """
        Calculate implantation receptivity
        Based on LIF, HOXA10, and other factors
        """
        key_genes = ['LIF', 'HOXA10', 'VEGFA']
        score = sum(gene_expression.get(g, 0) for g in key_genes) / len(key_genes)
        return min(100, score * 12)
    
    @staticmethod
    def mammary_development(gene_expression: Dict[str, float]) -> float:
        """
        Calculate mammary gland development score
        Based on RANKL, WNT4, and AREG
        """
        key_genes = ['RANKL', 'WNT4', 'AREG']
        score = sum(gene_expression.get(g, 0) for g in key_genes) / len(key_genes)
        return min(100, score * 15)
    
    @staticmethod
    def cell_cycle_progression(gene_expression: Dict[str, float]) -> float:
        """Calculate cell cycle progression rate"""
        cyclins = ['CCNA1', 'CCNB1', 'CCND1']
        cdks = ['CDK2', 'CDK4']
        
        cyclin_score = sum(gene_expression.get(g, 0) for g in cyclins)
        cdk_score = sum(gene_expression.get(g, 0) for g in cdks)
        
        return min(100, (cyclin_score + cdk_score) * 8)


class CrossTalkPathways:
    """Models crosstalk with other signaling pathways"""
    
    @staticmethod
    def estrogen_crosstalk(PR_level: float, ER_level: float = 100.0) -> float:
        """
        Estrogen receptor crosstalk
        ER upregulates PR expression
        """
        upregulation = ER_level * 0.02
        return PR_level * (1 + upregulation / 100)
    
    @staticmethod
    def growth_factor_crosstalk(PR_level: float, 
                                egf: float = 0, 
                                igf: float = 0) -> Dict[str, float]:
        """
        Crosstalk with growth factor signaling
        EGF and IGF can activate PR through phosphorylation
        """
        
        # Ligand-independent activation
        phospho_PR = (egf * 0.15 + igf * 0.12) * PR_level / 100
        
        return {
            'phosphorylated_PR': phospho_PR,
            'ligand_independent_activity': min(100, phospho_PR * 0.8)
        }
    
    @staticmethod
    def inflammatory_modulation(PR_DNA_bound: float) -> Dict[str, float]:
        """
        PR anti-inflammatory effects
        Suppresses NF-κB and promotes anti-inflammatory genes
        """
        
        nfkb_suppression = min(100, PR_DNA_bound * 0.6)
        il10_induction = min(100, PR_DNA_bound * 0.8)
        tgfb_induction = min(100, PR_DNA_bound * 0.7)
        
        return {
            'NFkB_suppression': nfkb_suppression,
            'IL10_induction': il10_induction,
            'TGFbeta_induction': tgfb_induction
        }
