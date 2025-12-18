"""
HGF/MET Signaling Pathway Simulator
====================================
Comprehensive simulation of Hepatocyte Growth Factor (HGF) and MET receptor signaling pathway
involved in cell motility, invasion, and liver regeneration.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Page configuration
st.set_page_config(
    page_title="HGF/MET Signaling Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PathwayParameters:
    """Parameters for HGF/MET signaling pathway"""
    # Ligand-Receptor Binding
    k_hgf_bind: float = 0.1      # HGF-MET binding rate
    k_hgf_unbind: float = 0.01   # HGF-MET unbinding rate
    k_met_deg: float = 0.05      # MET degradation rate
    k_met_synth: float = 0.1     # MET synthesis rate
    
    # Receptor Activation
    k_met_auto: float = 0.5      # MET autophosphorylation rate
    k_met_dephos: float = 0.2    # MET dephosphorylation rate
    
    # PI3K-AKT Pathway
    k_pi3k_act: float = 0.8      # PI3K activation by pMET
    k_pi3k_inact: float = 0.3    # PI3K inactivation
    k_akt_act: float = 1.0       # AKT activation by PI3K
    k_akt_inact: float = 0.4     # AKT inactivation
    
    # RAS-MAPK Pathway
    k_ras_act: float = 0.7       # RAS activation by pMET
    k_ras_inact: float = 0.35    # RAS inactivation
    k_erk_act: float = 0.9       # ERK activation by RAS
    k_erk_inact: float = 0.45    # ERK inactivation
    
    # STAT3 Pathway
    k_stat3_act: float = 0.6     # STAT3 activation by pMET
    k_stat3_inact: float = 0.25  # STAT3 inactivation
    
    # RAC1-CDC42 (Cell Motility)
    k_rac1_act: float = 0.75     # RAC1 activation
    k_rac1_inact: float = 0.3    # RAC1 inactivation
    k_cdc42_act: float = 0.65    # CDC42 activation
    k_cdc42_inact: float = 0.28  # CDC42 inactivation
    
    # Downstream Effects
    k_invasion: float = 0.5      # Invasion signal strength
    k_motility: float = 0.6      # Motility signal strength
    k_prolif: float = 0.4        # Proliferation signal strength
    k_survival: float = 0.55     # Survival signal strength
    
    # Negative Feedback
    k_feedback_akt: float = 0.1  # AKT feedback on MET
    k_feedback_erk: float = 0.08 # ERK feedback on SOS

class HGFMETModel:
    """Mathematical model of HGF/MET signaling pathway"""
    
    def __init__(self, params: PathwayParameters):
        self.params = params
        self.species_names = [
            'HGF', 'MET', 'HGF_MET', 'pMET',
            'PI3K', 'pPI3K', 'AKT', 'pAKT',
            'RAS', 'pRAS', 'ERK', 'pERK',
            'STAT3', 'pSTAT3',
            'RAC1', 'pRAC1', 'CDC42', 'pCDC42',
            'Invasion', 'Motility', 'Proliferation', 'Survival'
        ]
        
    def derivatives(self, state: np.ndarray, t: float, hgf_input: float) -> np.ndarray:
        """Calculate derivatives for ODE system"""
        (HGF, MET, HGF_MET, pMET,
         PI3K, pPI3K, AKT, pAKT,
         RAS, pRAS, ERK, pERK,
         STAT3, pSTAT3,
         RAC1, pRAC1, CDC42, pCDC42,
         Invasion, Motility, Proliferation, Survival) = state
        
        p = self.params
        
        # HGF and MET dynamics
        dHGF = hgf_input - p.k_hgf_bind * HGF * MET + p.k_hgf_unbind * HGF_MET
        dMET = p.k_met_synth - p.k_hgf_bind * HGF * MET + p.k_hgf_unbind * HGF_MET - p.k_met_deg * MET - p.k_feedback_akt * pAKT * MET
        dHGF_MET = p.k_hgf_bind * HGF * MET - p.k_hgf_unbind * HGF_MET - p.k_met_auto * HGF_MET
        dpMET = p.k_met_auto * HGF_MET - p.k_met_dephos * pMET - p.k_met_deg * pMET
        
        # PI3K-AKT pathway
        dpPI3K = p.k_pi3k_act * pMET * PI3K - p.k_pi3k_inact * pPI3K
        dPI3K = -dpPI3K
        dpAKT = p.k_akt_act * pPI3K * AKT - p.k_akt_inact * pAKT
        dAKT = -dpAKT
        
        # RAS-MAPK pathway
        feedback_erk = 1 / (1 + p.k_feedback_erk * pERK)
        dpRAS = p.k_ras_act * pMET * RAS * feedback_erk - p.k_ras_inact * pRAS
        dRAS = -dpRAS
        dpERK = p.k_erk_act * pRAS * ERK - p.k_erk_inact * pERK
        dERK = -dpERK
        
        # STAT3 pathway
        dpSTAT3 = p.k_stat3_act * pMET * STAT3 - p.k_stat3_inact * pSTAT3
        dSTAT3 = -dpSTAT3
        
        # RAC1-CDC42 (Cell motility)
        dpRAC1 = p.k_rac1_act * pMET * RAC1 - p.k_rac1_inact * pRAC1
        dRAC1 = -dpRAC1
        dpCDC42 = p.k_cdc42_act * pMET * CDC42 - p.k_cdc42_inact * pCDC42
        dCDC42 = -dpCDC42
        
        # Downstream cellular responses
        dInvasion = p.k_invasion * (pERK + pRAC1) - 0.1 * Invasion
        dMotility = p.k_motility * (pRAC1 + pCDC42) - 0.1 * Motility
        dProliferation = p.k_prolif * (pERK + pAKT) - 0.1 * Proliferation
        dSurvival = p.k_survival * (pAKT + pSTAT3) - 0.1 * Survival
        
        return np.array([
            dHGF, dMET, dHGF_MET, dpMET,
            dPI3K, dpPI3K, dAKT, dpAKT,
            dRAS, dpRAS, dERK, dpERK,
            dSTAT3, dpSTAT3,
            dRAC1, dpRAC1, dCDC42, dpCDC42,
            dInvasion, dMotility, dProliferation, dSurvival
        ])
    
    def get_initial_state(self) -> np.ndarray:
        """Get initial concentrations of all species"""
        return np.array([
            0.0,   # HGF
            10.0,  # MET
            0.0,   # HGF_MET
            0.0,   # pMET
            5.0,   # PI3K
            0.0,   # pPI3K
            8.0,   # AKT
            0.0,   # pAKT
            6.0,   # RAS
            0.0,   # pRAS
            7.0,   # ERK
            0.0,   # pERK
            5.0,   # STAT3
            0.0,   # pSTAT3
            4.0,   # RAC1
            0.0,   # pRAC1
            4.0,   # CDC42
            0.0,   # pCDC42
            0.0,   # Invasion
            0.0,   # Motility
            0.0,   # Proliferation
            0.0    # Survival
        ])
    
    def simulate(self, t_span: Tuple[float, float], hgf_protocol: str = 'constant',
                 hgf_amplitude: float = 1.0, pulse_duration: float = 10.0,
                 num_points: int = 1000) -> pd.DataFrame:
        """Simulate the pathway"""
        t = np.linspace(t_span[0], t_span[1], num_points)
        y0 = self.get_initial_state()
        
        # Define HGF input based on protocol
        def hgf_input(time):
            if hgf_protocol == 'constant':
                return hgf_amplitude if time > 0 else 0
            elif hgf_protocol == 'pulse':
                return hgf_amplitude if 0 < time < pulse_duration else 0
            elif hgf_protocol == 'ramp':
                return hgf_amplitude * min(time / 10.0, 1.0) if time > 0 else 0
            elif hgf_protocol == 'oscillatory':
                return hgf_amplitude * (1 + np.sin(2 * np.pi * time / 20.0)) / 2
            return 0
        
        # Integrate ODEs
        solution = odeint(self.derivatives, y0, t, args=(hgf_amplitude,))
        
        # Create dataframe
        df = pd.DataFrame(solution, columns=self.species_names)
        df['Time'] = t
        
        return df

def create_pathway_network():
    """Create network graph of HGF/MET pathway"""
    G = nx.DiGraph()
    
    # Add nodes with categories
    nodes = {
        'Ligand': ['HGF'],
        'Receptor': ['MET', 'pMET'],
        'PI3K-AKT': ['PI3K', 'pPI3K', 'AKT', 'pAKT'],
        'RAS-MAPK': ['RAS', 'pRAS', 'ERK', 'pERK'],
        'STAT3': ['STAT3', 'pSTAT3'],
        'Rho GTPases': ['RAC1', 'pRAC1', 'CDC42', 'pCDC42'],
        'Cellular Response': ['Invasion', 'Motility', 'Proliferation', 'Survival']
    }
    
    colors = {
        'Ligand': '#ff6b6b',
        'Receptor': '#4ecdc4',
        'PI3K-AKT': '#45b7d1',
        'RAS-MAPK': '#96ceb4',
        'STAT3': '#ffeaa7',
        'Rho GTPases': '#dfe6e9',
        'Cellular Response': '#a29bfe'
    }
    
    for category, node_list in nodes.items():
        for node in node_list:
            G.add_node(node, category=category, color=colors[category])
    
    # Add edges (activation arrows)
    edges = [
        ('HGF', 'MET', 'binding'),
        ('MET', 'pMET', 'activation'),
        ('pMET', 'pPI3K', 'activation'),
        ('pPI3K', 'pAKT', 'activation'),
        ('pMET', 'pRAS', 'activation'),
        ('pRAS', 'pERK', 'activation'),
        ('pMET', 'pSTAT3', 'activation'),
        ('pMET', 'pRAC1', 'activation'),
        ('pMET', 'pCDC42', 'activation'),
        ('pERK', 'Invasion', 'promotes'),
        ('pRAC1', 'Invasion', 'promotes'),
        ('pRAC1', 'Motility', 'promotes'),
        ('pCDC42', 'Motility', 'promotes'),
        ('pERK', 'Proliferation', 'promotes'),
        ('pAKT', 'Proliferation', 'promotes'),
        ('pAKT', 'Survival', 'promotes'),
        ('pSTAT3', 'Survival', 'promotes'),
        ('pAKT', 'MET', 'feedback'),
        ('pERK', 'RAS', 'feedback')
    ]
    
    for source, target, etype in edges:
        G.add_edge(source, target, type=etype)
    
    return G

def plot_network_graph(G):
    """Create interactive network visualization"""
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_type = edge[2]['type']
        color = 'red' if edge_type == 'feedback' else 'gray'
        width = 1 if edge_type == 'feedback' else 2
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(node[1]['color'])
        node_text.append(f"{node[0]}<br>Category: {node[1]['category']}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node[0] for node in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='HGF/MET Signaling Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    return fig

def plot_time_series(df, selected_species, title='Time Series'):
    """Create time series plot for selected species"""
    fig = go.Figure()
    
    for species in selected_species:
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df[species],
            mode='lines',
            name=species,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (min)',
        yaxis_title='Concentration (a.u.)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_heatmap(df, species_list):
    """Create heatmap of species concentrations over time"""
    data = df[species_list].T
    
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=df['Time'],
        y=species_list,
        colorscale='Viridis',
        hovertemplate='Time: %{x}<br>Species: %{y}<br>Concentration: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Pathway Activity Heatmap',
        xaxis_title='Time (min)',
        yaxis_title='Species',
        height=600,
        template='plotly_white'
    )
    
    return fig

def plot_phase_space(df, species_x, species_y):
    """Create phase space plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[species_x],
        y=df[species_y],
        mode='lines+markers',
        marker=dict(
            size=4,
            color=df['Time'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Time')
        ),
        line=dict(width=2, color='lightblue'),
        name='Trajectory'
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter(
        x=[df[species_x].iloc[0]],
        y=[df[species_y].iloc[0]],
        mode='markers',
        marker=dict(size=15, color='green', symbol='star'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter(
        x=[df[species_x].iloc[-1]],
        y=[df[species_y].iloc[-1]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='x'),
        name='End'
    ))
    
    fig.update_layout(
        title=f'Phase Space: {species_x} vs {species_y}',
        xaxis_title=species_x,
        yaxis_title=species_y,
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_dose_response(params, species, hgf_doses):
    """Create dose-response curve"""
    model = HGFMETModel(params)
    responses = []
    
    for dose in hgf_doses:
        df = model.simulate((0, 60), hgf_protocol='constant', hgf_amplitude=dose, num_points=500)
        # Get steady-state response (last value)
        responses.append(df[species].iloc[-1])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hgf_doses,
        y=responses,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f'Dose-Response Curve: {species}',
        xaxis_title='HGF Dose (a.u.)',
        yaxis_title=f'{species} Response (a.u.)',
        xaxis_type='log',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_cellular_responses(df):
    """Plot all cellular responses"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Invasion', 'Motility', 'Proliferation', 'Survival']
    )
    
    responses = ['Invasion', 'Motility', 'Proliferation', 'Survival']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
    
    for response, pos, color in zip(responses, positions, colors):
        fig.add_trace(
            go.Scatter(
                x=df['Time'],
                y=df[response],
                mode='lines',
                name=response,
                line=dict(width=3, color=color),
                fill='tozeroy'
            ),
            row=pos[0], col=pos[1]
        )
    
    fig.update_xaxes(title_text='Time (min)')
    fig.update_yaxes(title_text='Activity (a.u.)')
    fig.update_layout(
        title='Cellular Response Profiles',
        height=700,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_pathway_comparison(params_list, labels, species):
    """Compare multiple parameter sets"""
    fig = go.Figure()
    
    for params, label in zip(params_list, labels):
        model = HGFMETModel(params)
        df = model.simulate((0, 100), hgf_amplitude=1.0)
        
        fig.add_trace(go.Scatter(
            x=df['Time'],
            y=df[species],
            mode='lines',
            name=label,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f'Parameter Comparison: {species}',
        xaxis_title='Time (min)',
        yaxis_title='Concentration (a.u.)',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

# Main App
def main():
    st.markdown('<div class="main-header">HGF/MET Signaling Pathway Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive modeling of cell motility, invasion, and liver regeneration</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("Simulation Controls")
    
    # Simulation parameters
    st.sidebar.subheader("Time Settings")
    t_end = st.sidebar.slider("Simulation Duration (min)", 10, 200, 100)
    num_points = st.sidebar.slider("Time Points", 100, 2000, 1000)
    
    st.sidebar.subheader("HGF Stimulation")
    hgf_protocol = st.sidebar.selectbox(
        "Stimulation Protocol",
        ['constant', 'pulse', 'ramp', 'oscillatory']
    )
    hgf_amplitude = st.sidebar.slider("HGF Amplitude", 0.0, 5.0, 1.0, 0.1)
    
    if hgf_protocol == 'pulse':
        pulse_duration = st.sidebar.slider("Pulse Duration (min)", 1, 50, 10)
    else:
        pulse_duration = 10.0
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        st.write("**Receptor Dynamics**")
        k_hgf_bind = st.slider("HGF Binding Rate", 0.01, 1.0, 0.1, 0.01)
        k_met_auto = st.slider("MET Autophosphorylation", 0.1, 2.0, 0.5, 0.1)
        
        st.write("**PI3K-AKT Pathway**")
        k_pi3k_act = st.slider("PI3K Activation", 0.1, 2.0, 0.8, 0.1)
        k_akt_act = st.slider("AKT Activation", 0.1, 2.0, 1.0, 0.1)
        
        st.write("**RAS-MAPK Pathway**")
        k_ras_act = st.slider("RAS Activation", 0.1, 2.0, 0.7, 0.1)
        k_erk_act = st.slider("ERK Activation", 0.1, 2.0, 0.9, 0.1)
        
        st.write("**Cellular Responses**")
        k_invasion = st.slider("Invasion Rate", 0.1, 2.0, 0.5, 0.1)
        k_motility = st.slider("Motility Rate", 0.1, 2.0, 0.6, 0.1)
    
    # Create parameter object
    params = PathwayParameters(
        k_hgf_bind=k_hgf_bind,
        k_met_auto=k_met_auto,
        k_pi3k_act=k_pi3k_act,
        k_akt_act=k_akt_act,
        k_ras_act=k_ras_act,
        k_erk_act=k_erk_act,
        k_invasion=k_invasion,
        k_motility=k_motility
    )
    
    # Run simulation
    model = HGFMETModel(params)
    
    with st.spinner('Running simulation...'):
        df = model.simulate(
            (0, t_end),
            hgf_protocol=hgf_protocol,
            hgf_amplitude=hgf_amplitude,
            pulse_duration=pulse_duration,
            num_points=num_points
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Pathway Dynamics",
        "Network",
        "Analysis",
        "Dose-Response",
        "Comparisons"
    ])
    
    with tab1:
        st.header("Simulation Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_pmet = df['pMET'].max()
            st.metric("Max pMET", f"{max_pmet:.2f}", help="Peak MET phosphorylation")
        
        with col2:
            max_invasion = df['Invasion'].max()
            st.metric("Max Invasion", f"{max_invasion:.2f}", help="Peak invasion signal")
        
        with col3:
            max_motility = df['Motility'].max()
            st.metric("Max Motility", f"{max_motility:.2f}", help="Peak motility signal")
        
        with col4:
            max_prolif = df['Proliferation'].max()
            st.metric("Max Proliferation", f"{max_prolif:.2f}", help="Peak proliferation")
        
        st.markdown("---")
        
        # Cellular responses
        st.subheader("Cellular Response Profiles")
        fig_responses = plot_cellular_responses(df)
        st.plotly_chart(fig_responses, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        response_cols = ['Invasion', 'Motility', 'Proliferation', 'Survival']
        summary_data = {
            'Response': response_cols,
            'Peak': [df[col].max() for col in response_cols],
            'Mean': [df[col].mean() for col in response_cols],
            'Time to Peak (min)': [df['Time'][df[col].idxmax()] for col in response_cols]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.header("Pathway Dynamics")
        
        # Species selection
        pathway_group = st.selectbox(
            "Select Pathway",
            ['Receptor', 'PI3K-AKT', 'RAS-MAPK', 'STAT3', 'Rho GTPases', 'All Phosphorylated']
        )
        
        species_groups = {
            'Receptor': ['MET', 'HGF_MET', 'pMET'],
            'PI3K-AKT': ['PI3K', 'pPI3K', 'AKT', 'pAKT'],
            'RAS-MAPK': ['RAS', 'pRAS', 'ERK', 'pERK'],
            'STAT3': ['STAT3', 'pSTAT3'],
            'Rho GTPases': ['RAC1', 'pRAC1', 'CDC42', 'pCDC42'],
            'All Phosphorylated': ['pMET', 'pPI3K', 'pAKT', 'pRAS', 'pERK', 'pSTAT3', 'pRAC1', 'pCDC42']
        }
        
        selected_species = species_groups[pathway_group]
        
        # Time series plot
        fig_ts = plot_time_series(df, selected_species, f'{pathway_group} Time Series')
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Heatmap
        st.subheader("Activity Heatmap")
        fig_heatmap = plot_heatmap(df, selected_species)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.header("Pathway Network")
        
        st.info("Node colors: Blue = Upstream signaling | Green = MAPK pathway | Yellow = STAT3 | Purple = Cellular responses")
        
        G = create_pathway_network()
        fig_network = plot_network_graph(G)
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            st.metric("Total Edges", G.number_of_edges())
        with col3:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.metric("Avg Degree", f"{avg_degree:.2f}")
    
    with tab4:
        st.header("Advanced Analysis")
        
        # Phase space analysis
        st.subheader("Phase Space Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            species_x = st.selectbox("X-axis Species", model.species_names, index=model.species_names.index('pAKT'))
        with col2:
            species_y = st.selectbox("Y-axis Species", model.species_names, index=model.species_names.index('pERK'))
        
        fig_phase = plot_phase_space(df, species_x, species_y)
        st.plotly_chart(fig_phase, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Matrix")
        phospho_species = ['pMET', 'pPI3K', 'pAKT', 'pRAS', 'pERK', 'pSTAT3']
        corr_matrix = df[phospho_species].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig_corr.update_layout(
            title='Correlation Between Phosphorylated Species',
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab5:
        st.header("Dose-Response Analysis")
        
        st.info("Analyzing pathway response to varying HGF concentrations")
        
        dose_species = st.selectbox(
            "Select Output Species",
            ['pMET', 'pAKT', 'pERK', 'Invasion', 'Motility', 'Proliferation'],
            index=3
        )
        
        hgf_doses = np.logspace(-2, 1, 20)  # 0.01 to 10
        
        with st.spinner('Calculating dose-response...'):
            fig_dose = plot_dose_response(params, dose_species, hgf_doses)
        
        st.plotly_chart(fig_dose, use_container_width=True)
        
        # EC50 analysis
        st.subheader("Sensitivity Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Response Metrics**")
            responses = []
            for dose in [0.1, 0.5, 1.0, 2.0, 5.0]:
                temp_df = model.simulate((0, 60), hgf_amplitude=dose, num_points=500)
                responses.append({
                    'HGF Dose': dose,
                    f'{dose_species}': temp_df[dose_species].iloc[-1]
                })
            st.dataframe(pd.DataFrame(responses))
        
        with col2:
            st.write("**Dynamic Range**")
            min_response = min([r[dose_species] for r in responses])
            max_response = max([r[dose_species] for r in responses])
            dynamic_range = max_response / min_response if min_response > 0 else float('inf')
            st.metric("Dynamic Range", f"{dynamic_range:.2f}x")
    
    with tab6:
        st.header("Parameter Comparisons")
        
        st.info("Compare different parameter sets or conditions")
        
        comparison_type = st.selectbox(
            "Comparison Type",
            ['Normal vs Enhanced PI3K', 'Normal vs Enhanced MAPK', 'Normal vs Feedback Inhibited']
        )
        
        # Create parameter sets
        params_normal = params
        
        if comparison_type == 'Normal vs Enhanced PI3K':
            params_modified = PathwayParameters(
                k_hgf_bind=params.k_hgf_bind,
                k_met_auto=params.k_met_auto,
                k_pi3k_act=params.k_pi3k_act * 2,  # Enhanced
                k_akt_act=params.k_akt_act * 2,
                k_ras_act=params.k_ras_act,
                k_erk_act=params.k_erk_act
            )
            labels = ['Normal', 'Enhanced PI3K']
            species_to_compare = 'pAKT'
            
        elif comparison_type == 'Normal vs Enhanced MAPK':
            params_modified = PathwayParameters(
                k_hgf_bind=params.k_hgf_bind,
                k_met_auto=params.k_met_auto,
                k_pi3k_act=params.k_pi3k_act,
                k_akt_act=params.k_akt_act,
                k_ras_act=params.k_ras_act * 2,  # Enhanced
                k_erk_act=params.k_erk_act * 2
            )
            labels = ['Normal', 'Enhanced MAPK']
            species_to_compare = 'pERK'
            
        else:  # Feedback Inhibited
            params_modified = PathwayParameters(
                k_hgf_bind=params.k_hgf_bind,
                k_met_auto=params.k_met_auto,
                k_pi3k_act=params.k_pi3k_act,
                k_akt_act=params.k_akt_act,
                k_ras_act=params.k_ras_act,
                k_erk_act=params.k_erk_act,
                k_feedback_akt=0.0,  # No feedback
                k_feedback_erk=0.0
            )
            labels = ['Normal', 'No Feedback']
            species_to_compare = 'pMET'
        
        fig_comp = plot_pathway_comparison(
            [params_normal, params_modified],
            labels,
            species_to_compare
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Show all cellular responses comparison
        st.subheader("Cellular Response Comparison")
        
        model_normal = HGFMETModel(params_normal)
        model_modified = HGFMETModel(params_modified)
        
        df_normal = model_normal.simulate((0, 100), hgf_amplitude=1.0)
        df_modified = model_modified.simulate((0, 100), hgf_amplitude=1.0)
        
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Invasion', 'Motility', 'Proliferation', 'Survival']
        )
        
        responses = ['Invasion', 'Motility', 'Proliferation', 'Survival']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for response, pos in zip(responses, positions):
            fig_multi.add_trace(
                go.Scatter(x=df_normal['Time'], y=df_normal[response],
                          name=f'{labels[0]}', line=dict(dash='solid')),
                row=pos[0], col=pos[1]
            )
            fig_multi.add_trace(
                go.Scatter(x=df_modified['Time'], y=df_modified[response],
                          name=f'{labels[1]}', line=dict(dash='dash')),
                row=pos[0], col=pos[1]
            )
        
        fig_multi.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig_multi, use_container_width=True)
    
    # Data export
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Data")
    
    if st.sidebar.button("Download Simulation Data"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"hgf_met_simulation_{hgf_protocol}.csv",
            mime="text/csv"
        )
    
    # Information footer
    with st.expander("About HGF/MET Signaling"):
        st.markdown("""
        ### Hepatocyte Growth Factor (HGF) / MET Receptor Signaling
        
        **Biological Context:**
        - HGF is a pleiotropic cytokine that binds to its receptor MET
        - Critical for embryonic development, wound healing, and tissue regeneration
        - Dysregulation implicated in cancer metastasis and invasion
        
        **Key Pathways:**
        1. **PI3K-AKT**: Cell survival, metabolism, growth
        2. **RAS-MAPK**: Proliferation, differentiation
        3. **STAT3**: Gene transcription, survival
        4. **RAC1/CDC42**: Cytoskeletal reorganization, cell motility
        
        **Cellular Outcomes:**
        - **Invasion**: Matrix degradation, cell migration through barriers
        - **Motility**: Cell movement, lamellipodia formation
        - **Proliferation**: Cell cycle progression
        - **Survival**: Anti-apoptotic signaling
        
        **Model Features:**
        - Ordinary differential equations (ODEs)
        - Mass-action kinetics
        - Negative feedback loops
        - Multiple timescale dynamics
        """)

if __name__ == "__main__":
    main()
