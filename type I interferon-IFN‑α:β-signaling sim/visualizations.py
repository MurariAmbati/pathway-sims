"""
Advanced visualization tools for IFN-α/β signaling pathway.

Includes time series plots, phase portraits, network diagrams,
heatmaps, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_pathway_timecourse(t: np.ndarray, 
                            solution: np.ndarray,
                            state_names: List[str],
                            components: List[str] = None) -> go.Figure:
    """
    Create interactive time course plots for pathway components.
    
    Args:
        t: Time array
        solution: Solution array from simulation
        state_names: List of state variable names
        components: Specific components to plot (None = plot all)
        
    Returns:
        Plotly figure object
    """
    if components is None:
        components = ['IFN_IFNAR', 'JAK_active', 'pSTAT1', 'pSTAT2', 
                     'ISGF3_nuc', 'ISG_mRNA', 'ISG_protein', 'SOCS']
    
    fig = go.Figure()
    
    for comp in components:
        if comp in state_names:
            idx = state_names.index(comp)
            fig.add_trace(go.Scatter(
                x=t,
                y=solution[:, idx],
                mode='lines',
                name=comp,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title='Type I IFN Signaling Pathway Dynamics',
        xaxis_title='Time (minutes)',
        yaxis_title='Concentration (molecules/cell)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def plot_signaling_cascade(t: np.ndarray,
                           solution: np.ndarray,
                           state_names: List[str]) -> go.Figure:
    """
    Create subplot showing the signaling cascade progression.
    
    Args:
        t: Time array
        solution: Solution array from simulation
        state_names: List of state variable names
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Receptor Activation',
            'JAK Activation',
            'STAT Phosphorylation',
            'ISGF3 Formation',
            'ISG Expression',
            'Negative Feedback'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Receptor activation
    ifn_ifnar_idx = state_names.index('IFN_IFNAR')
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, ifn_ifnar_idx], 
                  name='IFN-IFNAR', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # JAK activation
    jak_active_idx = state_names.index('JAK_active')
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, jak_active_idx],
                  name='JAK Active', line=dict(color='#ff7f0e')),
        row=1, col=2
    )
    
    # STAT phosphorylation
    pstat1_idx = state_names.index('pSTAT1')
    pstat2_idx = state_names.index('pSTAT2')
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, pstat1_idx],
                  name='pSTAT1', line=dict(color='#2ca02c')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, pstat2_idx],
                  name='pSTAT2', line=dict(color='#d62728')),
        row=2, col=1
    )
    
    # ISGF3 formation
    isgf3_cyto_idx = state_names.index('ISGF3_cyto')
    isgf3_nuc_idx = state_names.index('ISGF3_nuc')
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, isgf3_cyto_idx],
                  name='ISGF3 (Cytoplasm)', line=dict(color='#9467bd')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, isgf3_nuc_idx],
                  name='ISGF3 (Nucleus)', line=dict(color='#8c564b')),
        row=2, col=2
    )
    
    # ISG expression
    isg_mrna_idx = state_names.index('ISG_mRNA')
    isg_protein_idx = state_names.index('ISG_protein')
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, isg_mrna_idx],
                  name='ISG mRNA', line=dict(color='#e377c2')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, isg_protein_idx],
                  name='ISG Protein', line=dict(color='#7f7f7f')),
        row=3, col=1
    )
    
    # Negative feedback
    socs_idx = state_names.index('SOCS')
    fig.add_trace(
        go.Scatter(x=t, y=solution[:, socs_idx],
                  name='SOCS', line=dict(color='#bcbd22')),
        row=3, col=2
    )
    
    fig.update_xaxes(title_text="Time (min)", row=3, col=1)
    fig.update_xaxes(title_text="Time (min)", row=3, col=2)
    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Concentration", row=1, col=2)
    fig.update_yaxes(title_text="Concentration", row=2, col=1)
    fig.update_yaxes(title_text="Concentration", row=2, col=2)
    fig.update_yaxes(title_text="Concentration", row=3, col=1)
    fig.update_yaxes(title_text="Concentration", row=3, col=2)
    
    fig.update_layout(
        title_text="IFN-α/β Signaling Cascade Progression",
        height=900,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_phase_portrait(solution: np.ndarray,
                       state_names: List[str],
                       x_component: str,
                       y_component: str,
                       time_points: np.ndarray = None) -> go.Figure:
    """
    Create phase portrait of two pathway components.
    
    Args:
        solution: Solution array from simulation
        state_names: List of state variable names
        x_component: Name of x-axis component
        y_component: Name of y-axis component
        time_points: Optional time array for color coding
        
    Returns:
        Plotly figure object
    """
    x_idx = state_names.index(x_component)
    y_idx = state_names.index(y_component)
    
    x_data = solution[:, x_idx]
    y_data = solution[:, y_idx]
    
    if time_points is not None:
        fig = go.Figure(data=go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers+lines',
            marker=dict(
                size=4,
                color=time_points,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time (min)")
            ),
            line=dict(width=1, color='rgba(100,100,100,0.3)'),
            name='Trajectory'
        ))
    else:
        fig = go.Figure(data=go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            marker=dict(size=4),
            name='Trajectory'
        ))
    
    # Mark start and end points
    fig.add_trace(go.Scatter(
        x=[x_data[0]],
        y=[y_data[0]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='star'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_data[-1]],
        y=[y_data[-1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        name='End'
    ))
    
    fig.update_layout(
        title=f'Phase Portrait: {y_component} vs {x_component}',
        xaxis_title=x_component,
        yaxis_title=y_component,
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_pathway_network(highlight_active: bool = True) -> go.Figure:
    """
    Create network diagram of the IFN signaling pathway.
    
    Args:
        highlight_active: Whether to highlight active components
        
    Returns:
        Plotly figure object
    """
    # Define pathway network structure
    G = nx.DiGraph()
    
    # Add nodes
    nodes = {
        'IFN-α/β': {'layer': 0, 'color': '#1f77b4'},
        'IFNAR1/2': {'layer': 1, 'color': '#ff7f0e'},
        'JAK1': {'layer': 2, 'color': '#2ca02c'},
        'TYK2': {'layer': 2, 'color': '#2ca02c'},
        'STAT1': {'layer': 3, 'color': '#d62728'},
        'STAT2': {'layer': 3, 'color': '#d62728'},
        'IRF9': {'layer': 4, 'color': '#9467bd'},
        'ISGF3': {'layer': 4, 'color': '#8c564b'},
        'ISGs': {'layer': 5, 'color': '#e377c2'},
        'SOCS': {'layer': 6, 'color': '#bcbd22'},
        'Antiviral': {'layer': 7, 'color': '#17becf'}
    }
    
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)
    
    # Add edges (interactions)
    edges = [
        ('IFN-α/β', 'IFNAR1/2'),
        ('IFNAR1/2', 'JAK1'),
        ('IFNAR1/2', 'TYK2'),
        ('JAK1', 'STAT1'),
        ('JAK1', 'STAT2'),
        ('TYK2', 'STAT1'),
        ('TYK2', 'STAT2'),
        ('STAT1', 'ISGF3'),
        ('STAT2', 'ISGF3'),
        ('IRF9', 'ISGF3'),
        ('ISGF3', 'ISGs'),
        ('ISGs', 'Antiviral'),
        ('ISGF3', 'SOCS'),
        ('SOCS', 'JAK1'),  # Negative feedback
        ('SOCS', 'TYK2'),  # Negative feedback
        ('ISGs', 'IFN-α/β'),  # Positive feedback
    ]
    
    G.add_edges_from(edges)
    
    # Calculate layout
    pos = {}
    layers = {}
    for node, attrs in nodes.items():
        layer = attrs['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
    
    for layer, layer_nodes in layers.items():
        n_nodes = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            x = layer * 2
            y = (i - (n_nodes - 1) / 2) * 2
            pos[node] = (x, y)
    
    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Check if feedback edge
        is_feedback = (edge[0] in ['SOCS', 'ISGs'] and edge[1] in ['JAK1', 'TYK2', 'IFN-α/β'])
        
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=2 if not is_feedback else 1.5,
                color='#888' if not is_feedback else '#ff4444',
                dash='solid' if not is_feedback else 'dash'
            ),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(nodes[node]['color'])
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='white', family='Arial Black'),
        hoverinfo='text',
        marker=dict(
            size=40,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title='Type I Interferon Signaling Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_parameter_sensitivity_heatmap(sensitivity_data: pd.DataFrame) -> go.Figure:
    """
    Create heatmap showing parameter sensitivity analysis results.
    
    Args:
        sensitivity_data: DataFrame with parameters and their effects
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_data.values,
        x=sensitivity_data.columns,
        y=sensitivity_data.index,
        colorscale='RdBu_r',
        zmid=0,
        text=sensitivity_data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Sensitivity")
    ))
    
    fig.update_layout(
        title='Parameter Sensitivity Analysis',
        xaxis_title='Output Metrics',
        yaxis_title='Parameters',
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_dose_response(doses: np.ndarray,
                      responses: np.ndarray,
                      metric_name: str = 'ISG Expression') -> go.Figure:
    """
    Create dose-response curve for IFN concentration.
    
    Args:
        doses: Array of IFN doses
        responses: Array of response values
        metric_name: Name of the measured response
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=doses,
        y=responses,
        mode='markers+lines',
        marker=dict(size=10, color='#1f77b4'),
        line=dict(width=2, color='#1f77b4'),
        name=metric_name
    ))
    
    fig.update_layout(
        title=f'IFN Dose-Response Curve: {metric_name}',
        xaxis_title='IFN Concentration (molecules/cell)',
        yaxis_title=metric_name,
        xaxis_type='log',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_comparative_timecourse(t_list: List[np.ndarray],
                               solution_list: List[np.ndarray],
                               state_names: List[str],
                               component: str,
                               labels: List[str]) -> go.Figure:
    """
    Compare time courses under different conditions.
    
    Args:
        t_list: List of time arrays
        solution_list: List of solution arrays
        state_names: List of state variable names
        component: Component to compare
        labels: Labels for each condition
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    idx = state_names.index(component)
    
    colors = px.colors.qualitative.Plotly
    
    for i, (t, solution, label) in enumerate(zip(t_list, solution_list, labels)):
        fig.add_trace(go.Scatter(
            x=t,
            y=solution[:, idx],
            mode='lines',
            name=label,
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=f'Comparative Analysis: {component}',
        xaxis_title='Time (minutes)',
        yaxis_title=f'{component} Concentration',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig
