"""
Visualization utilities for progesterone receptor signaling
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import pandas as pd


class ReceptorVisualizer:
    """Visualization for receptor dynamics"""
    
    def __init__(self):
        sns.set_style("whitegrid")
        self.colors = {
            'PR_A': '#2E86AB',
            'PR_B': '#A23B72',
            'complex': '#F18F01',
            'nuclear': '#C73E1D',
            'protein': '#6A994E'
        }
    
    def plot_receptor_states(self, time: np.ndarray, 
                            solution: np.ndarray) -> go.Figure:
        """Plot receptor state transitions over time"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cytoplasmic Receptor Dynamics',
                'Nuclear Receptor Translocation',
                'DNA Binding and Transcription',
                'Protein Expression'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Cytoplasmic receptors
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 0], name='PR-A (unbound)',
                      line=dict(color=self.colors['PR_A'], width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 1], name='PR-B (unbound)',
                      line=dict(color=self.colors['PR_B'], width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 2], name='PR-A:P4',
                      line=dict(color=self.colors['PR_A'], width=2, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 3], name='PR-B:P4',
                      line=dict(color=self.colors['PR_B'], width=2, dash='dash')),
            row=1, col=1
        )
        
        # Nuclear translocation
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 4], name='PR-A dimer (cyto)',
                      line=dict(color=self.colors['PR_A'], width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 5], name='PR-B dimer (cyto)',
                      line=dict(color=self.colors['PR_B'], width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 6], name='PR-A dimer (nuclear)',
                      line=dict(color=self.colors['nuclear'], width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 7], name='PR-B dimer (nuclear)',
                      line=dict(color=self.colors['nuclear'], width=2, dash='dash')),
            row=1, col=2
        )
        
        # DNA binding
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 8], name='PR:DNA',
                      line=dict(color=self.colors['complex'], width=3)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 9], name='PR:DNA:Coactivator',
                      line=dict(color=self.colors['nuclear'], width=3)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 10], name='mRNA',
                      line=dict(color='#06A77D', width=2, dash='dot')),
            row=2, col=1
        )
        
        # Protein expression
        fig.add_trace(
            go.Scatter(x=time, y=solution[:, 11], name='Target Protein',
                      line=dict(color=self.colors['protein'], width=3),
                      fill='tozeroy'),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (min)", row=1, col=1)
        fig.update_xaxes(title_text="Time (min)", row=1, col=2)
        fig.update_xaxes(title_text="Time (min)", row=2, col=1)
        fig.update_xaxes(title_text="Time (min)", row=2, col=2)
        
        fig.update_yaxes(title_text="Concentration (nM)", row=1, col=1)
        fig.update_yaxes(title_text="Concentration (nM)", row=1, col=2)
        fig.update_yaxes(title_text="Concentration (nM)", row=2, col=1)
        fig.update_yaxes(title_text="Concentration (nM)", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            title_text="Progesterone Receptor Signaling Cascade",
            title_x=0.5
        )
        
        return fig
    
    def plot_dose_response(self, concentrations: np.ndarray,
                          responses: np.ndarray,
                          metric_name: str = 'Response') -> go.Figure:
        """Plot dose-response curve"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=concentrations,
            y=responses,
            mode='lines+markers',
            marker=dict(size=10, color=self.colors['complex']),
            line=dict(width=3, color=self.colors['nuclear'])
        ))
        
        fig.update_layout(
            title=f'Dose-Response Curve: {metric_name}',
            xaxis_title='Progesterone Concentration (nM)',
            xaxis_type='log',
            yaxis_title=metric_name,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_heatmap_gene_expression(self, gene_data: pd.DataFrame) -> go.Figure:
        """Plot heatmap of gene expression across conditions"""
        
        fig = go.Figure(data=go.Heatmap(
            z=gene_data.values,
            x=gene_data.columns,
            y=gene_data.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Expression Level")
        ))
        
        fig.update_layout(
            title='Gene Expression Heatmap',
            xaxis_title='Conditions',
            yaxis_title='Genes',
            template='plotly_white',
            height=600
        )
        
        return fig


class PathwayVisualizer:
    """Visualization for signaling pathways"""
    
    @staticmethod
    def plot_pathway_activation(pathway_data: Dict[str, float]) -> go.Figure:
        """Plot pathway activation levels"""
        
        pathways = list(pathway_data.keys())
        values = list(pathway_data.values())
        
        fig = go.Figure(go.Bar(
            x=values,
            y=pathways,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Activation Level")
            )
        ))
        
        fig.update_layout(
            title='Signaling Pathway Activation',
            xaxis_title='Activation Level',
            yaxis_title='Pathway',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_cellular_responses(response_data: Dict[str, float]) -> go.Figure:
        """Plot cellular response metrics"""
        
        categories = list(response_data.keys())
        values = list(response_data.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='#C73E1D', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Cellular Response Profile',
            template='plotly_white',
            height=500
        )
        
        return fig


class ComparisonVisualizer:
    """Visualization for comparing conditions"""
    
    @staticmethod
    def plot_tissue_comparison(tissue_data: Dict[str, np.ndarray],
                              time: np.ndarray,
                              metric: str = 'Nuclear Receptor') -> go.Figure:
        """Compare signaling across tissue types"""
        
        fig = go.Figure()
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (tissue, data) in enumerate(tissue_data.items()):
            fig.add_trace(go.Scatter(
                x=time,
                y=data,
                name=tissue,
                line=dict(width=3, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=f'{metric} Across Tissue Types',
            xaxis_title='Time (min)',
            yaxis_title=f'{metric} (nM)',
            template='plotly_white',
            height=500,
            legend=dict(x=0.7, y=0.95)
        )
        
        return fig
    
    @staticmethod
    def plot_trimester_comparison(trimester_data: Dict[int, np.ndarray],
                                 time: np.ndarray) -> go.Figure:
        """Compare signaling across pregnancy trimesters"""
        
        fig = go.Figure()
        
        colors = {1: '#6A994E', 2: '#F18F01', 3: '#C73E1D'}
        
        for trimester, data in trimester_data.items():
            fig.add_trace(go.Scatter(
                x=time,
                y=data,
                name=f'Trimester {trimester}',
                line=dict(width=3, color=colors[trimester]),
                fill='tonexty' if trimester > 1 else None
            ))
        
        fig.update_layout(
            title='Progesterone Signaling Across Pregnancy',
            xaxis_title='Time (min)',
            yaxis_title='Response (nM)',
            template='plotly_white',
            height=500
        )
        
        return fig


class NetworkVisualizer:
    """Visualization for signaling networks"""
    
    @staticmethod
    def create_network_diagram() -> go.Figure:
        """Create interactive network diagram of signaling cascade"""
        
        # Define nodes
        nodes = {
            'P4': (0, 5),
            'PR-A': (1, 7),
            'PR-B': (1, 3),
            'PR:P4': (2, 5),
            'PR Dimer': (3, 5),
            'Nuclear PR': (4, 5),
            'PRE': (5, 5),
            'Coactivators': (5, 7),
            'mRNA': (6, 5),
            'Protein': (7, 5),
            'MAPK': (3, 8),
            'PI3K': (3, 2),
        }
        
        # Create edges
        edges = [
            ('P4', 'PR-A'), ('P4', 'PR-B'),
            ('PR-A', 'PR:P4'), ('PR-B', 'PR:P4'),
            ('PR:P4', 'PR Dimer'),
            ('PR Dimer', 'Nuclear PR'),
            ('Nuclear PR', 'PRE'),
            ('Coactivators', 'PRE'),
            ('PRE', 'mRNA'),
            ('mRNA', 'Protein'),
            ('PR:P4', 'MAPK'),
            ('PR:P4', 'PI3K'),
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for source, target in edges:
            x0, y0 = nodes[source]
            x1, y1 = nodes[target]
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        x_nodes = [pos[0] for pos in nodes.values()]
        y_nodes = [pos[1] for pos in nodes.values()]
        labels = list(nodes.keys())
        
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
            text=labels,
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Progesterone Receptor Signaling Network',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            height=600,
            width=800
        )
        
        return fig
