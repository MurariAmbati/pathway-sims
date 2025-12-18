"""
Visualization utilities for FcγR signaling simulations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import networkx as nx


# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#262730'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['grid.color'] = '#444444'


class SignalingVisualizer:
    """Visualization tools for signaling pathway data"""
    
    @staticmethod
    def plot_signaling_cascade(data: Dict[str, np.ndarray], 
                               time_points: np.ndarray,
                               title: str = "FcγR Signaling Cascade") -> go.Figure:
        """
        Create interactive plot of signaling cascade
        
        Args:
            data: Dictionary of signaling components
            time_points: Time points array
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Receptor Binding & ITAM',
                'Syk Activation',
                'PI3K/Akt Pathway',
                'MAPK Cascade',
                'Calcium Signaling',
                'Transcription Factors',
                'Cytokine Production',
                'Inhibitory Signaling'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Receptor & ITAM
        fig.add_trace(
            go.Scatter(x=time_points, y=data['bound_receptors'], 
                      name='Bound FcγR', line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['phospho_itam'], 
                      name='p-ITAM', line=dict(color='#4ECDC4')),
            row=1, col=1
        )
        
        # Syk
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_syk'], 
                      name='Active Syk', line=dict(color='#95E1D3')),
            row=1, col=2
        )
        
        # PI3K/Akt
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_pi3k'], 
                      name='Active PI3K', line=dict(color='#F38181')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_akt'], 
                      name='Active Akt', line=dict(color='#AA96DA')),
            row=2, col=1
        )
        
        # MAPK
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_mek'], 
                      name='Active MEK', line=dict(color='#FCBAD3')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_erk'], 
                      name='Active ERK', line=dict(color='#FFFFD2')),
            row=2, col=2
        )
        
        # Calcium
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_plcg'], 
                      name='Active PLCγ', line=dict(color='#A8E6CF')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['calcium'], 
                      name='Ca²⁺', line=dict(color='#FFD3B6')),
            row=3, col=1
        )
        
        # Transcription factors
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_nfat'], 
                      name='Active NFAT', line=dict(color='#FFAAA5')),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_nfkb'], 
                      name='Active NF-κB', line=dict(color='#FF8B94')),
            row=3, col=2
        )
        
        # Cytokines
        fig.add_trace(
            go.Scatter(x=time_points, y=data['tnf_alpha'], 
                      name='TNF-α', line=dict(color='#C7CEEA')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['il6'], 
                      name='IL-6', line=dict(color='#B5EAD7')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['il1_beta'], 
                      name='IL-1β', line=dict(color='#FFDAC1')),
            row=4, col=1
        )
        
        # Inhibitory
        fig.add_trace(
            go.Scatter(x=time_points, y=data['active_ship'], 
                      name='Active SHIP', line=dict(color='#FF9AA2', dash='dash')),
            row=4, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Activity/Concentration (normalized)")
        
        fig.update_layout(
            height=1200,
            title_text=title,
            showlegend=True,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_pathway_network(include_inhibitory: bool = False) -> go.Figure:
        """
        Create network diagram of FcγR signaling pathway
        
        Args:
            include_inhibitory: Include inhibitory pathways
            
        Returns:
            Plotly figure
        """
        G = nx.DiGraph()
        
        # Define nodes
        nodes = [
            ('FcγR', 0),
            ('ITAM', 1),
            ('Syk', 2),
            ('PI3K', 3),
            ('Akt', 3),
            ('PLCγ', 3),
            ('MEK', 3),
            ('IP3', 4),
            ('Ca²⁺', 4),
            ('ERK', 4),
            ('NFAT', 5),
            ('NF-κB', 5),
            ('Cytokines', 6)
        ]
        
        if include_inhibitory:
            nodes.extend([
                ('FcγRIIB', 0),
                ('SHIP', 2)
            ])
        
        # Add nodes
        for node, layer in nodes:
            G.add_node(node, layer=layer)
        
        # Define edges (activating)
        edges_activating = [
            ('FcγR', 'ITAM'),
            ('ITAM', 'Syk'),
            ('Syk', 'PI3K'),
            ('Syk', 'PLCγ'),
            ('Syk', 'MEK'),
            ('PI3K', 'Akt'),
            ('PLCγ', 'IP3'),
            ('IP3', 'Ca²⁺'),
            ('MEK', 'ERK'),
            ('Ca²⁺', 'NFAT'),
            ('ERK', 'NF-κB'),
            ('NFAT', 'Cytokines'),
            ('NF-κB', 'Cytokines'),
            ('Akt', 'Cytokines')
        ]
        
        edges_inhibitory = []
        if include_inhibitory:
            edges_inhibitory = [
                ('FcγRIIB', 'SHIP'),
                ('SHIP', 'PI3K')
            ]
        
        G.add_edges_from(edges_activating)
        G.add_edges_from(edges_inhibitory)
        
        # Layout
        pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
        
        # Create edge traces
        edge_trace_activating = []
        edge_trace_inhibitory = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            if edge in edges_inhibitory:
                edge_trace_inhibitory.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='#FF6B6B', dash='dash'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
            else:
                edge_trace_activating.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=2, color='#4ECDC4'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        color_map = {
            0: '#FF6B6B',  # Receptor
            1: '#4ECDC4',  # ITAM
            2: '#95E1D3',  # Kinases
            3: '#F38181',  # Secondary messengers
            4: '#AA96DA',  # Tertiary
            5: '#FCBAD3',  # TFs
            6: '#FFD93D'   # Output
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_colors.append(color_map[G.nodes[node]['layer']])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            textfont=dict(size=12, color='white')
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace_activating + edge_trace_inhibitory + [node_trace])
        
        fig.update_layout(
            title='FcγR Signaling Network',
            showlegend=False,
            hovermode='closest',
            template='plotly_dark',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_phagocytosis_kinetics(data: Dict[str, np.ndarray],
                                   time_points: np.ndarray) -> go.Figure:
        """Plot phagocytosis kinetics"""
        fig = go.Figure()
        
        colors = {
            'free_targets': '#FF6B6B',
            'bound_targets': '#4ECDC4',
            'engulfed_targets': '#95E1D3',
            'digested_targets': '#F38181',
            'total_phagocytosed': '#FFD93D'
        }
        
        labels = {
            'free_targets': 'Free Targets',
            'bound_targets': 'Bound Targets',
            'engulfed_targets': 'Engulfed',
            'digested_targets': 'Digested',
            'total_phagocytosed': 'Total Phagocytosed'
        }
        
        for key in ['free_targets', 'bound_targets', 'engulfed_targets', 
                    'digested_targets', 'total_phagocytosed']:
            if key in data:
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=data[key],
                    mode='lines',
                    name=labels[key],
                    line=dict(width=3, color=colors[key])
                ))
        
        fig.update_layout(
            title='Phagocytosis Kinetics',
            xaxis_title='Time (minutes)',
            yaxis_title='Fraction of Targets',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_cytotoxicity_kinetics(data: Dict[str, np.ndarray],
                                   time_points: np.ndarray) -> go.Figure:
        """Plot ADCC kinetics"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cell States', 'Specific Lysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cell states
        fig.add_trace(
            go.Scatter(x=time_points, y=data['viable_targets'],
                      name='Viable Targets', line=dict(color='#4ECDC4', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['conjugated'],
                      name='Conjugated', line=dict(color='#FFD93D', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['dead_targets'],
                      name='Dead Targets', line=dict(color='#FF6B6B', width=3)),
            row=1, col=1
        )
        
        # Specific lysis
        fig.add_trace(
            go.Scatter(x=time_points, y=data['specific_lysis'],
                      name='Specific Lysis (%)', line=dict(color='#F38181', width=4),
                      fill='tozeroy'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
        fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
        fig.update_yaxes(title_text="Fraction", row=1, col=1)
        fig.update_yaxes(title_text="Lysis (%)", row=1, col=2)
        
        fig.update_layout(
            title='Antibody-Dependent Cellular Cytotoxicity (ADCC)',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_dose_response(antibody_conc: np.ndarray,
                          efficiency: np.ndarray,
                          ylabel: str = "Efficiency") -> go.Figure:
        """Plot dose-response curve"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=antibody_conc,
            y=efficiency,
            mode='lines+markers',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8, color='#FF6B6B')
        ))
        
        # Add EC50 line if applicable
        if efficiency.max() > 0.5:
            ec50_idx = np.argmin(np.abs(efficiency - 0.5))
            ec50 = antibody_conc[ec50_idx]
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text=f"EC50 = {ec50:.2f} µg/mL")
            fig.add_vline(x=ec50, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Dose-Response Curve',
            xaxis_title='Antibody Concentration (µg/mL)',
            yaxis_title=ylabel,
            template='plotly_dark',
            xaxis_type='log',
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_clustering_dynamics(data: Dict[str, np.ndarray],
                                time_points: np.ndarray) -> go.Figure:
        """Plot FcγR clustering dynamics"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cluster Distribution', 'Average Cluster Size')
        )
        
        # Cluster distribution
        fig.add_trace(
            go.Scatter(x=time_points, y=data['monomers'],
                      name='Monomers', line=dict(color='#4ECDC4')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['dimers'],
                      name='Dimers', line=dict(color='#95E1D3')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['trimers'],
                      name='Trimers', line=dict(color='#F38181')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=data['large_clusters'],
                      name='Large Clusters', line=dict(color='#FFD93D')),
            row=1, col=1
        )
        
        # Average size
        fig.add_trace(
            go.Scatter(x=time_points, y=data['average_cluster_size'],
                      name='Avg Size', line=dict(color='#FF6B6B', width=3)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Fraction", row=1, col=1)
        fig.update_yaxes(title_text="Receptors per Cluster", row=1, col=2)
        
        fig.update_layout(
            title='FcγR Clustering Dynamics',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        return fig


class ComparisonVisualizer:
    """Tools for comparing different conditions"""
    
    @staticmethod
    def compare_cell_types(data_dict: Dict[str, Dict],
                          time_points: np.ndarray,
                          metric: str = 'specific_lysis') -> go.Figure:
        """Compare different cell types or conditions"""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#FFD93D']
        
        for idx, (label, data) in enumerate(data_dict.items()):
            if metric in data:
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=data[metric],
                    mode='lines',
                    name=label,
                    line=dict(width=3, color=colors[idx % len(colors)])
                ))
        
        fig.update_layout(
            title=f'Comparison: {metric.replace("_", " ").title()}',
            xaxis_title='Time',
            yaxis_title=metric.replace("_", " ").title(),
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(data: pd.DataFrame, 
                      title: str = "Parameter Sensitivity") -> go.Figure:
        """Create heatmap for parameter sensitivity analysis"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlBu_r',
            text=data.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark',
            height=600
        )
        
        return fig
