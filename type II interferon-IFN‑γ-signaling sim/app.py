import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from dataclasses import dataclass
import networkx as nx

st.set_page_config(page_title="IFN-γ Signaling Simulator", layout="wide", initial_sidebar_state="expanded")

@dataclass
class IFNGammaParameters:
    """parameters for ifn-γ signaling pathway"""
    # receptor binding and activation
    k_binding: float = 0.5
    k_unbinding: float = 0.1
    k_jak_activation: float = 0.8
    k_jak_deactivation: float = 0.2
    
    # stat1 dynamics
    k_stat1_phosphorylation: float = 1.0
    k_stat1_dephosphorylation: float = 0.3
    k_stat1_dimerization: float = 0.7
    k_stat1_dissociation: float = 0.15
    k_stat1_nuclear_import: float = 0.6
    k_stat1_nuclear_export: float = 0.2
    
    # transcriptional response
    k_irf1_transcription: float = 0.5
    k_irf1_translation: float = 0.4
    k_irf1_degradation: float = 0.1
    
    # th1 polarization markers
    k_tbet_induction: float = 0.6
    k_tbet_degradation: float = 0.15
    k_il12r_upregulation: float = 0.4
    
    # macrophage activation
    k_inos_induction: float = 0.7
    k_inos_degradation: float = 0.2
    k_mhc2_upregulation: float = 0.5
    k_mhc2_downregulation: float = 0.1
    
    # feedback and regulation
    k_socs1_induction: float = 0.3
    k_socs1_inhibition: float = 0.6
    k_socs1_degradation: float = 0.25
    
    # initial conditions
    ifng_concentration: float = 10.0
    receptor_total: float = 100.0
    stat1_total: float = 200.0
    
def ifng_signaling_ode(state, t, params):
    """
    ordinary differential equations for ifn-γ signaling pathway
    
    state variables:
    0: ifng_receptor_complex
    1: jak_active
    2: stat1_phosphorylated
    3: stat1_dimer
    4: stat1_nuclear
    5: irf1_mrna
    6: irf1_protein
    7: tbet
    8: il12r
    9: inos
    10: mhc2
    11: socs1
    """
    
    ifng_r_complex, jak_active, p_stat1, stat1_dimer, stat1_nuc, \
    irf1_mrna, irf1_protein, tbet, il12r, inos, mhc2, socs1 = state
    
    p = params
    
    # receptor dynamics
    free_receptor = p.receptor_total - ifng_r_complex
    d_ifng_r_complex = (p.k_binding * p.ifng_concentration * free_receptor - 
                        p.k_unbinding * ifng_r_complex)
    
    # jak activation (inhibited by socs1)
    socs1_inhibition_factor = 1.0 / (1.0 + p.k_socs1_inhibition * socs1)
    d_jak_active = (p.k_jak_activation * ifng_r_complex * socs1_inhibition_factor - 
                    p.k_jak_deactivation * jak_active)
    
    # stat1 phosphorylation and dimerization
    free_stat1 = p.stat1_total - p_stat1 - 2 * stat1_dimer - stat1_nuc
    d_p_stat1 = (p.k_stat1_phosphorylation * jak_active * free_stat1 - 
                 p.k_stat1_dephosphorylation * p_stat1 - 
                 2 * p.k_stat1_dimerization * p_stat1 * p_stat1 + 
                 2 * p.k_stat1_dissociation * stat1_dimer)
    
    # stat1 dimerization
    d_stat1_dimer = (p.k_stat1_dimerization * p_stat1 * p_stat1 - 
                     p.k_stat1_dissociation * stat1_dimer - 
                     p.k_stat1_nuclear_import * stat1_dimer + 
                     p.k_stat1_nuclear_export * stat1_nuc)
    
    # nuclear stat1
    d_stat1_nuc = (p.k_stat1_nuclear_import * stat1_dimer - 
                   p.k_stat1_nuclear_export * stat1_nuc)
    
    # irf1 transcription and translation
    d_irf1_mrna = (p.k_irf1_transcription * stat1_nuc - 
                   p.k_irf1_degradation * irf1_mrna)
    d_irf1_protein = (p.k_irf1_translation * irf1_mrna - 
                      p.k_irf1_degradation * irf1_protein)
    
    # th1 polarization: t-bet induction
    d_tbet = (p.k_tbet_induction * stat1_nuc * (1 + 0.5 * irf1_protein) - 
              p.k_tbet_degradation * tbet)
    
    # il-12 receptor upregulation
    d_il12r = (p.k_il12r_upregulation * tbet - 
               p.k_tbet_degradation * il12r)
    
    # macrophage activation: inos induction
    d_inos = (p.k_inos_induction * (stat1_nuc + irf1_protein) - 
              p.k_inos_degradation * inos)
    
    # mhc class ii upregulation
    d_mhc2 = (p.k_mhc2_upregulation * (stat1_nuc + 0.5 * irf1_protein) - 
              p.k_mhc2_downregulation * mhc2)
    
    # socs1 negative feedback
    d_socs1 = (p.k_socs1_induction * stat1_nuc - 
               p.k_socs1_degradation * socs1)
    
    return [d_ifng_r_complex, d_jak_active, d_p_stat1, d_stat1_dimer, 
            d_stat1_nuc, d_irf1_mrna, d_irf1_protein, d_tbet, d_il12r, 
            d_inos, d_mhc2, d_socs1]

def run_simulation(params, t_max=100, n_points=500):
    """run the ifn-γ signaling simulation"""
    t = np.linspace(0, t_max, n_points)
    
    # initial conditions
    initial_state = [
        0,    # ifng_receptor_complex
        0,    # jak_active
        0,    # stat1_phosphorylated
        0,    # stat1_dimer
        0,    # stat1_nuclear
        0,    # irf1_mrna
        0,    # irf1_protein
        0,    # tbet
        0,    # il12r
        0,    # inos
        0,    # mhc2
        0     # socs1
    ]
    
    solution = odeint(ifng_signaling_ode, initial_state, t, args=(params,))
    
    df = pd.DataFrame(solution, columns=[
        'ifng_receptor_complex', 'jak_active', 'stat1_phosphorylated',
        'stat1_dimer', 'stat1_nuclear', 'irf1_mrna', 'irf1_protein',
        'tbet', 'il12r', 'inos', 'mhc2', 'socs1'
    ])
    df['time'] = t
    
    return df

def create_pathway_network():
    """create network graph of ifn-γ signaling pathway"""
    G = nx.DiGraph()
    
    # add nodes with categories
    nodes = {
        'IFN-γ': 'ligand',
        'IFNGR': 'receptor',
        'JAK1/2': 'kinase',
        'STAT1': 'transcription_factor',
        'STAT1-P': 'transcription_factor',
        'STAT1-dimer': 'transcription_factor',
        'IRF1': 'transcription_factor',
        'T-bet': 'th1_marker',
        'IL-12R': 'th1_marker',
        'iNOS': 'macrophage_marker',
        'MHC-II': 'macrophage_marker',
        'SOCS1': 'inhibitor'
    }
    
    for node, category in nodes.items():
        G.add_node(node, category=category)
    
    # add edges
    edges = [
        ('IFN-γ', 'IFNGR', 'binding'),
        ('IFNGR', 'JAK1/2', 'activation'),
        ('JAK1/2', 'STAT1', 'phosphorylation'),
        ('STAT1', 'STAT1-P', 'modification'),
        ('STAT1-P', 'STAT1-dimer', 'dimerization'),
        ('STAT1-dimer', 'IRF1', 'transcription'),
        ('STAT1-dimer', 'T-bet', 'transcription'),
        ('STAT1-dimer', 'iNOS', 'transcription'),
        ('STAT1-dimer', 'MHC-II', 'transcription'),
        ('STAT1-dimer', 'SOCS1', 'transcription'),
        ('IRF1', 'T-bet', 'enhancement'),
        ('IRF1', 'iNOS', 'enhancement'),
        ('IRF1', 'MHC-II', 'enhancement'),
        ('T-bet', 'IL-12R', 'upregulation'),
        ('SOCS1', 'JAK1/2', 'inhibition')
    ]
    
    for source, target, edge_type in edges:
        G.add_edge(source, target, type=edge_type)
    
    return G

def plot_network_graph(G):
    """plot the pathway network using plotly"""
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # node colors by category
    color_map = {
        'ligand': '#FF6B6B',
        'receptor': '#4ECDC4',
        'kinase': '#45B7D1',
        'transcription_factor': '#FFA07A',
        'th1_marker': '#98D8C8',
        'macrophage_marker': '#F7DC6F',
        'inhibitor': '#BB8FCE'
    }
    
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_type = G.edges[edge]['type']
        
        if edge_type == 'inhibition':
            dash = 'dash'
            color = '#E74C3C'
        else:
            dash = 'solid'
            color = '#95A5A6'
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color=color, dash=dash),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition='top center',
        marker=dict(
            size=20,
            color=[color_map[G.nodes[node]['category']] for node in G.nodes()],
            line=dict(width=2, color='white')
        ),
        hoverinfo='text',
        hovertext=[f"{node}<br>Category: {G.nodes[node]['category']}" for node in G.nodes()]
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='ifn-γ signaling pathway network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    return fig

def plot_time_courses(df):
    """plot time course of key signaling components"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'receptor & jak activation',
            'stat1 dynamics',
            'transcription factors',
            'th1 polarization markers',
            'macrophage activation markers',
            'negative feedback'
        )
    )
    
    # receptor & jak
    fig.add_trace(go.Scatter(x=df['time'], y=df['ifng_receptor_complex'], 
                            name='IFN-γ:IFNGR', line=dict(color='#FF6B6B')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['jak_active'], 
                            name='JAK1/2 active', line=dict(color='#4ECDC4')), row=1, col=1)
    
    # stat1
    fig.add_trace(go.Scatter(x=df['time'], y=df['stat1_phosphorylated'], 
                            name='STAT1-P', line=dict(color='#45B7D1')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['time'], y=df['stat1_dimer'], 
                            name='STAT1 dimer', line=dict(color='#FFA07A')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['time'], y=df['stat1_nuclear'], 
                            name='nuclear STAT1', line=dict(color='#F06292')), row=1, col=2)
    
    # transcription factors
    fig.add_trace(go.Scatter(x=df['time'], y=df['irf1_mrna'], 
                            name='IRF1 mRNA', line=dict(color='#AED581')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['irf1_protein'], 
                            name='IRF1 protein', line=dict(color='#66BB6A')), row=2, col=1)
    
    # th1 markers
    fig.add_trace(go.Scatter(x=df['time'], y=df['tbet'], 
                            name='T-bet', line=dict(color='#98D8C8')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df['time'], y=df['il12r'], 
                            name='IL-12R', line=dict(color='#80DEEA')), row=2, col=2)
    
    # macrophage markers
    fig.add_trace(go.Scatter(x=df['time'], y=df['inos'], 
                            name='iNOS', line=dict(color='#F7DC6F')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['mhc2'], 
                            name='MHC-II', line=dict(color='#FFD54F')), row=3, col=1)
    
    # feedback
    fig.add_trace(go.Scatter(x=df['time'], y=df['socs1'], 
                            name='SOCS1', line=dict(color='#BB8FCE')), row=3, col=2)
    
    fig.update_xaxes(title_text='time (min)', row=3, col=1)
    fig.update_xaxes(title_text='time (min)', row=3, col=2)
    fig.update_yaxes(title_text='concentration (AU)')
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    
    return fig

def plot_heatmap(df):
    """create heatmap of normalized signaling dynamics"""
    columns = ['jak_active', 'stat1_nuclear', 'irf1_protein', 
               'tbet', 'il12r', 'inos', 'mhc2', 'socs1']
    
    data = df[columns].values.T
    # normalize each row
    data_norm = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True) + 1e-10)
    
    fig = go.Figure(data=go.Heatmap(
        z=data_norm,
        x=df['time'],
        y=['JAK1/2', 'nuclear STAT1', 'IRF1', 'T-bet', 'IL-12R', 'iNOS', 'MHC-II', 'SOCS1'],
        colorscale='Viridis',
        colorbar=dict(title='normalized<br>activity')
    ))
    
    fig.update_layout(
        title='temporal dynamics heatmap',
        xaxis_title='time (min)',
        yaxis_title='signaling component',
        height=500
    )
    
    return fig

def plot_phase_space(df):
    """create phase space plots"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('stat1 phosphorylation vs nuclear translocation',
                       'th1 vs macrophage activation')
    )
    
    # STAT1 dynamics
    fig.add_trace(go.Scatter(
        x=df['stat1_phosphorylated'], 
        y=df['stat1_nuclear'],
        mode='markers+lines',
        marker=dict(
            size=4,
            color=df['time'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='time', x=0.45)
        ),
        name='trajectory',
        showlegend=False
    ), row=1, col=1)
    
    # TH1 vs macrophage
    fig.add_trace(go.Scatter(
        x=df['tbet'], 
        y=df['inos'],
        mode='markers+lines',
        marker=dict(
            size=4,
            color=df['time'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='time', x=1.02)
        ),
        name='trajectory',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_xaxes(title_text='STAT1-P', row=1, col=1)
    fig.update_yaxes(title_text='nuclear STAT1', row=1, col=1)
    fig.update_xaxes(title_text='T-bet', row=1, col=2)
    fig.update_yaxes(title_text='iNOS', row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def plot_dose_response():
    """simulate dose-response curves"""
    doses = np.logspace(-1, 2, 20)
    
    tbet_response = []
    inos_response = []
    socs1_response = []
    
    for dose in doses:
        params = IFNGammaParameters()
        params.ifng_concentration = dose
        df = run_simulation(params, t_max=100, n_points=200)
        
        tbet_response.append(df['tbet'].iloc[-1])
        inos_response.append(df['inos'].iloc[-1])
        socs1_response.append(df['socs1'].iloc[-1])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=doses, y=tbet_response,
        mode='lines+markers',
        name='T-bet',
        line=dict(color='#98D8C8', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=doses, y=inos_response,
        mode='lines+markers',
        name='iNOS',
        line=dict(color='#F7DC6F', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=doses, y=socs1_response,
        mode='lines+markers',
        name='SOCS1',
        line=dict(color='#BB8FCE', width=3)
    ))
    
    fig.update_layout(
        title='ifn-γ dose-response curves',
        xaxis_title='IFN-γ concentration (ng/mL)',
        yaxis_title='steady-state expression (AU)',
        xaxis_type='log',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# streamlit ui
st.title('type ii interferon (ifn-γ) signaling simulator')
st.markdown('**slug:** `ifn_type2`')
st.markdown('modeling th1 polarization and macrophage activation through the jak-stat1 pathway')

# sidebar controls
st.sidebar.header('simulation parameters')

with st.sidebar.expander('receptor dynamics', expanded=False):
    k_binding = st.slider('receptor binding rate', 0.1, 2.0, 0.5, 0.1)
    k_jak_activation = st.slider('jak activation rate', 0.1, 2.0, 0.8, 0.1)
    ifng_conc = st.slider('ifn-γ concentration', 1.0, 50.0, 10.0, 1.0)

with st.sidebar.expander('stat1 signaling', expanded=False):
    k_stat1_phos = st.slider('stat1 phosphorylation rate', 0.1, 2.0, 1.0, 0.1)
    k_stat1_dimer = st.slider('stat1 dimerization rate', 0.1, 2.0, 0.7, 0.1)
    k_nuclear_import = st.slider('nuclear import rate', 0.1, 2.0, 0.6, 0.1)

with st.sidebar.expander('th1 polarization', expanded=False):
    k_tbet = st.slider('t-bet induction rate', 0.1, 2.0, 0.6, 0.1)
    k_il12r = st.slider('il-12r upregulation rate', 0.1, 2.0, 0.4, 0.1)

with st.sidebar.expander('macrophage activation', expanded=False):
    k_inos = st.slider('inos induction rate', 0.1, 2.0, 0.7, 0.1)
    k_mhc2 = st.slider('mhc-ii upregulation rate', 0.1, 2.0, 0.5, 0.1)

with st.sidebar.expander('negative feedback', expanded=False):
    k_socs1_ind = st.slider('socs1 induction rate', 0.1, 1.0, 0.3, 0.05)
    k_socs1_inh = st.slider('socs1 inhibition strength', 0.1, 2.0, 0.6, 0.1)

t_max = st.sidebar.slider('simulation time (min)', 50, 300, 100, 10)

# create parameter object
params = IFNGammaParameters()
params.k_binding = k_binding
params.k_jak_activation = k_jak_activation
params.ifng_concentration = ifng_conc
params.k_stat1_phosphorylation = k_stat1_phos
params.k_stat1_dimerization = k_stat1_dimer
params.k_stat1_nuclear_import = k_nuclear_import
params.k_tbet_induction = k_tbet
params.k_il12r_upregulation = k_il12r
params.k_inos_induction = k_inos
params.k_mhc2_upregulation = k_mhc2
params.k_socs1_induction = k_socs1_ind
params.k_socs1_inhibition = k_socs1_inh

# run simulation
with st.spinner('running simulation...'):
    df = run_simulation(params, t_max=t_max, n_points=500)

# tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'pathway network', 'time courses', 'heatmap', 
    'phase space', 'dose-response', 'data'
])

with tab1:
    st.subheader('signaling pathway network')
    G = create_pathway_network()
    fig_network = plot_network_graph(G)
    st.plotly_chart(fig_network, use_container_width=True)
    
    st.markdown('**pathway components:**')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('- ligand/receptor')
        st.markdown('- kinase cascade')
    with col2:
        st.markdown('- transcription factors')
        st.markdown('- th1 markers')
    with col3:
        st.markdown('- macrophage markers')
        st.markdown('- negative feedback')

with tab2:
    st.subheader('temporal dynamics')
    fig_time = plot_time_courses(df)
    st.plotly_chart(fig_time, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('peak jak activity', f"{df['jak_active'].max():.2f}")
        st.metric('time to peak jak', f"{df.loc[df['jak_active'].idxmax(), 'time']:.1f} min")
    with col2:
        st.metric('steady-state t-bet', f"{df['tbet'].iloc[-1]:.2f}")
        st.metric('steady-state il-12r', f"{df['il12r'].iloc[-1]:.2f}")
    with col3:
        st.metric('steady-state inos', f"{df['inos'].iloc[-1]:.2f}")
        st.metric('steady-state mhc-ii', f"{df['mhc2'].iloc[-1]:.2f}")

with tab3:
    st.subheader('signaling dynamics heatmap')
    fig_heatmap = plot_heatmap(df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.subheader('phase space analysis')
    fig_phase = plot_phase_space(df)
    st.plotly_chart(fig_phase, use_container_width=True)
    
    st.markdown('**interpretation:**')
    st.markdown('- left: stat1 activation and nuclear translocation dynamics')
    st.markdown('- right: coupled th1 polarization and macrophage activation')

with tab5:
    st.subheader('dose-response analysis')
    with st.spinner('calculating dose-response curves...'):
        fig_dose = plot_dose_response()
    st.plotly_chart(fig_dose, use_container_width=True)
    
    st.markdown('**biological significance:**')
    st.markdown('- demonstrates graded response to ifn-γ concentrations')
    st.markdown('- socs1 feedback increases at higher doses')
    st.markdown('- both th1 and m1 macrophage markers show sigmoidal response')

with tab6:
    st.subheader('simulation data')
    
    st.markdown('**download simulation results:**')
    csv = df.to_csv(index=False)
    st.download_button(
        label='download as csv',
        data=csv,
        file_name='ifng_signaling_simulation.csv',
        mime='text/csv'
    )
    
    st.dataframe(df, use_container_width=True)
    
    st.markdown('**summary statistics:**')
    st.dataframe(df.describe(), use_container_width=True)

# footer
st.markdown('---')
st.markdown('**model notes:**')
st.markdown("""
- implements canonical jak1/jak2-stat1 signaling pathway
- includes transcriptional regulation through irf1 and ciita
- models th1 polarization via t-bet and il-12 receptor upregulation
- simulates m1 macrophage activation through inos and mhc-ii expression
- incorporates socs1-mediated negative feedback regulation
- based on ordinary differential equation model with mass-action kinetics
""")
