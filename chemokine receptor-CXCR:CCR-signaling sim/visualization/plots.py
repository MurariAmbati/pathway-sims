"""
visualization utilities for plots and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
import seaborn as sns


def plot_gradient_2d(gradient_field: np.ndarray, 
                    slice_axis: int = 2,
                    slice_index: int = None,
                    title: str = "chemokine gradient") -> plt.Figure:
    """
    plot 2d slice of 3d gradient
    
    args:
        gradient_field: 3d concentration array
        slice_axis: axis to slice (0=x, 1=y, 2=z)
        slice_index: index along slice axis (default: middle)
        title: plot title
    
    returns:
        matplotlib figure
    """
    if slice_index is None:
        slice_index = gradient_field.shape[slice_axis] // 2
    
    if slice_axis == 0:
        data = gradient_field[slice_index, :, :]
    elif slice_axis == 1:
        data = gradient_field[:, slice_index, :]
    else:
        data = gradient_field[:, :, slice_index]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data.T, origin='lower', cmap='viridis', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('x (grid points)')
    ax.set_ylabel('y (grid points)')
    
    plt.colorbar(im, ax=ax, label='concentration')
    
    return fig


def plot_trajectories_3d(trajectories: List[np.ndarray],
                         colors: List[str] = None,
                         title: str = "cell trajectories") -> go.Figure:
    """
    plot 3d trajectories using plotly
    
    args:
        trajectories: list of trajectory arrays (n_timepoints, 3)
        colors: list of colors for each trajectory
        title: plot title
    
    returns:
        plotly figure
    """
    fig = go.Figure()
    
    n_traj = len(trajectories)
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_traj))
        colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
                 for c in colors]
    
    for i, traj in enumerate(trajectories):
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines+markers',
            marker=dict(size=2),
            line=dict(width=2, color=colors[i]),
            name=f'cell {i}',
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (μm)',
            yaxis_title='y (μm)',
            zaxis_title='z (μm)',
            aspectmode='data'
        ),
        height=700
    )
    
    return fig


def plot_receptor_dynamics(time: np.ndarray,
                          receptor_states: Dict[str, np.ndarray],
                          title: str = "receptor dynamics") -> go.Figure:
    """
    plot receptor state populations over time
    
    args:
        time: time array
        receptor_states: dict mapping state names to time series
        title: plot title
    
    returns:
        plotly figure
    """
    fig = go.Figure()
    
    colors = {
        'free': '#3498db',
        'bound': '#e74c3c',
        'desensitized': '#f39c12',
        'internalized': '#9b59b6'
    }
    
    for state_name, values in receptor_states.items():
        fig.add_trace(go.Scatter(
            x=time,
            y=values,
            mode='lines',
            name=state_name,
            line=dict(width=2, color=colors.get(state_name, '#95a5a6'))
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='time (s)',
        yaxis_title='receptor count',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_signaling_pathways(time: np.ndarray,
                           pathway_states: Dict[str, np.ndarray],
                           title: str = "signaling pathways") -> go.Figure:
    """
    plot signaling pathway activation over time
    
    args:
        time: time array
        pathway_states: dict of pathway component time series
        title: plot title
    
    returns:
        plotly figure with subplots
    """
    # group pathways
    pi3k_components = ['pi3k_active', 'pip3', 'akt_active']
    plc_components = ['plc_active', 'ip3', 'ca_cyto']
    mapk_components = ['mek_active', 'erk_active']
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('pi3k/akt pathway', 'plc/calcium pathway', 'mapk/erk pathway'),
        vertical_spacing=0.12
    )
    
    # pi3k pathway
    for component in pi3k_components:
        if component in pathway_states:
            fig.add_trace(
                go.Scatter(x=time, y=pathway_states[component], 
                          name=component, mode='lines'),
                row=1, col=1
            )
    
    # plc pathway
    for component in plc_components:
        if component in pathway_states:
            fig.add_trace(
                go.Scatter(x=time, y=pathway_states[component],
                          name=component, mode='lines'),
                row=2, col=1
            )
    
    # mapk pathway
    for component in mapk_components:
        if component in pathway_states:
            fig.add_trace(
                go.Scatter(x=time, y=pathway_states[component],
                          name=component, mode='lines'),
                row=3, col=1
            )
    
    fig.update_xaxes(title_text='time (s)', row=3, col=1)
    fig.update_yaxes(title_text='concentration (au)')
    
    fig.update_layout(height=900, title_text=title, hovermode='x unified')
    
    return fig


def plot_population_statistics(results: Dict) -> go.Figure:
    """
    plot population-level statistics
    
    args:
        results: simulation results dict
    
    returns:
        plotly figure with multiple subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('mean speed', 'state distribution', 
                       'displacement distribution', 'chemotactic index'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    time = results['time']
    
    # mean speed
    fig.add_trace(
        go.Scatter(x=time, y=results['mean_speeds'], 
                  mode='lines', name='mean speed',
                  line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    
    # state distribution (final)
    final_states = results['states'][-1]
    state_counts = {s: final_states.count(s) for s in set(final_states)}
    
    fig.add_trace(
        go.Bar(x=list(state_counts.keys()), y=list(state_counts.values()),
              marker_color='#2ecc71'),
        row=1, col=2
    )
    
    # displacement distribution
    initial_pos = results['positions'][0]
    final_pos = results['positions'][-1]
    displacements = np.linalg.norm(final_pos - initial_pos, axis=1)
    
    fig.add_trace(
        go.Histogram(x=displacements, nbinsx=20, marker_color='#e74c3c'),
        row=2, col=1
    )
    
    # chemotactic index (if available)
    if 'chemotactic_indices' in results and len(results['chemotactic_indices']) > 0:
        fig.add_trace(
            go.Scatter(x=time, y=results['chemotactic_indices'],
                      mode='lines', name='CI',
                      line=dict(color='#9b59b6', width=2)),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text='time (s)', row=1, col=1)
    fig.update_yaxes(title_text='speed (μm/s)', row=1, col=1)
    
    fig.update_xaxes(title_text='state', row=1, col=2)
    fig.update_yaxes(title_text='count', row=1, col=2)
    
    fig.update_xaxes(title_text='displacement (μm)', row=2, col=1)
    fig.update_yaxes(title_text='frequency', row=2, col=1)
    
    fig.update_xaxes(title_text='time (s)', row=2, col=2)
    fig.update_yaxes(title_text='chemotactic index', row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    
    return fig


def create_heatmap_animation(gradient_snapshots: List[np.ndarray],
                            time_points: np.ndarray,
                            slice_axis: int = 2) -> FuncAnimation:
    """
    create animated heatmap of gradient evolution
    
    args:
        gradient_snapshots: list of 3d gradient arrays
        time_points: corresponding time points
        slice_axis: axis to slice
    
    returns:
        matplotlib animation
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # initial frame
    slice_idx = gradient_snapshots[0].shape[slice_axis] // 2
    if slice_axis == 2:
        data = gradient_snapshots[0][:, :, slice_idx]
    else:
        data = gradient_snapshots[0][slice_idx, :, :]
    
    im = ax.imshow(data.T, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='concentration')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                       color='white', fontsize=12)
    
    def update(frame):
        if slice_axis == 2:
            data = gradient_snapshots[frame][:, :, slice_idx]
        else:
            data = gradient_snapshots[frame][slice_idx, :, :]
        
        im.set_array(data.T)
        time_text.set_text(f't = {time_points[frame]:.1f} s')
        return [im, time_text]
    
    anim = FuncAnimation(fig, update, frames=len(gradient_snapshots),
                        interval=100, blit=True)
    
    return anim
