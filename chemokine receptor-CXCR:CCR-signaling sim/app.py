"""
streamlit web application for chemokine receptor signaling simulation
interactive visualization and parameter control
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from models import (
    create_neutrophil_receptors,
    create_t_cell_receptors,
    create_monocyte_receptors,
    NEUTROPHIL_PROPS,
    T_CELL_PROPS,
    MONOCYTE_PROPS,
    RECEPTOR_LIBRARY,
    CHEMOKINE_LIBRARY
)

from simulation import (
    create_neutrophil_recruitment_simulation,
    create_t_cell_homing_simulation
)


# page configuration
st.set_page_config(
    page_title="chemokine receptor signaling",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# custom css
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """main application"""
    
    # header
    st.markdown('<div class="main-header">chemokine receptor (cxcr/ccr) signaling</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">leukocyte trafficking and positioning simulation</div>',
                unsafe_allow_html=True)
    
    # sidebar controls
    with st.sidebar:
        st.header("simulation controls")
        
        # scenario selection
        scenario = st.selectbox(
            "scenario",
            ["neutrophil recruitment", "t-cell homing", "custom"]
        )
        
        # cell type
        if scenario == "custom":
            cell_type = st.selectbox(
                "cell type",
                ["neutrophil", "t-cell", "monocyte"]
            )
        
        # parameters
        st.subheader("parameters")
        
        n_cells = st.slider("number of cells", 10, 100, 50)
        duration = st.slider("duration (s)", 60, 600, 300)
        
        # receptor configuration
        with st.expander("receptor configuration"):
            st.write("**expressed receptors**")
            
            if scenario == "neutrophil recruitment" or \
               (scenario == "custom" and cell_type == "neutrophil"):
                cxcr1_expr = st.slider("cxcr1 expression", 0.0, 2.0, 1.2)
                cxcr2_expr = st.slider("cxcr2 expression", 0.0, 2.0, 1.5)
            elif scenario == "t-cell homing" or \
                 (scenario == "custom" and cell_type == "t-cell"):
                cxcr4_expr = st.slider("cxcr4 expression", 0.0, 2.0, 1.0)
                ccr7_expr = st.slider("ccr7 expression", 0.0, 2.0, 1.2)
        
        # run button
        run_simulation = st.button("▶ run simulation", type="primary")
    
    # main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "overview",
        "live tracking",
        "analysis",
        "receptor dynamics",
        "documentation"
    ])
    
    # tab 1: overview
    with tab1:
        display_overview()
    
    # tab 2: live tracking
    with tab2:
        if run_simulation:
            run_live_simulation(scenario, n_cells, duration)
        else:
            st.info("configure parameters and click 'run simulation' to start")
    
    # tab 3: analysis
    with tab3:
        if 'simulation_results' in st.session_state:
            display_analysis(st.session_state['simulation_results'])
        else:
            st.info("run simulation first to see analysis")
    
    # tab 4: receptor dynamics
    with tab4:
        display_receptor_info()
    
    # tab 5: documentation
    with tab5:
        display_documentation()


def display_overview():
    """overview panel"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### chemokine receptors")
        st.markdown("""
        - **cxcr family**: cxcr1-7
        - **ccr family**: ccr1-10
        - gpcr signaling
        - gradient sensing
        - receptor dynamics
        """)
    
    with col2:
        st.markdown("### leukocyte trafficking")
        st.markdown("""
        - rolling adhesion
        - firm arrest
        - transmigration
        - directed migration
        - chemotaxis
        """)
    
    with col3:
        st.markdown("### signal transduction")
        st.markdown("""
        - g-protein activation
        - pi3k/akt pathway
        - plc/calcium release
        - mapk/erk cascade
        - rac/cdc42 gtpases
        """)
    
    st.markdown("---")
    
    # system architecture
    st.subheader("system architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**spatial scales**")
        st.markdown("""
        - molecular: receptor-ligand binding
        - cellular: signal transduction, migration
        - tissue: gradient formation, population dynamics
        """)
    
    with col2:
        st.markdown("**temporal scales**")
        st.markdown("""
        - ms: ligand binding
        - seconds: signaling cascades
        - minutes: cell migration
        - hours: tissue infiltration
        """)


def run_live_simulation(scenario: str, n_cells: int, duration: float):
    """run simulation with live updates"""
    
    st.subheader(f"{scenario}")
    
    # create simulation
    with st.spinner("initializing simulation..."):
        if scenario == "neutrophil recruitment":
            engine = create_neutrophil_recruitment_simulation(n_cells, duration)
        elif scenario == "t-cell homing":
            engine = create_t_cell_homing_simulation(n_cells, duration)
        else:
            engine = create_neutrophil_recruitment_simulation(n_cells, duration)
    
    # progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # placeholders for live plots
    col1, col2 = st.columns(2)
    
    with col1:
        plot_3d = st.empty()
    
    with col2:
        plot_metrics = st.empty()
    
    # callback for updates
    update_interval = 20  # update every 20 time points
    update_counter = 0
    
    def progress_callback(sim_time, progress):
        nonlocal update_counter
        update_counter += 1
        
        progress_bar.progress(progress)
        status_text.text(f"time: {sim_time:.1f} s / {duration:.1f} s")
        
        if update_counter % update_interval == 0:
            # get current state
            positions = engine.population.get_positions()
            states = engine.population.get_states()
            
            # 3d scatter plot
            fig_3d = create_3d_cell_plot(positions, states)
            plot_3d.plotly_chart(fig_3d, use_container_width=True)
            
            # metrics plot
            fig_metrics = create_metrics_plot(
                engine.state.time_points,
                engine.state.mean_speeds
            )
            plot_metrics.plotly_chart(fig_metrics, use_container_width=True)
    
    # run simulation
    start_time = time.time()
    engine.run(progress_callback=progress_callback)
    elapsed_time = time.time() - start_time
    
    # completion
    status_text.text(f"✅ simulation complete ({elapsed_time:.2f}s)")
    
    # final visualization
    results = engine.get_results()
    st.session_state['simulation_results'] = results
    
    # summary statistics
    st.markdown("---")
    st.subheader("summary statistics")
    
    stats = engine.get_summary_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("cells", stats['total_cells'])
    
    with col2:
        st.metric("mean displacement", f"{stats['mean_displacement']:.1f} μm")
    
    with col3:
        st.metric("mean speed", f"{stats['mean_speed']:.2f} μm/s")
    
    with col4:
        st.metric("timesteps", stats['timesteps'])


def create_3d_cell_plot(positions: np.ndarray, states: list) -> go.Figure:
    """create 3d scatter plot of cell positions"""
    
    # color by state
    state_colors = {
        'migrating': '#3498db',
        'arrested': '#e74c3c',
        'rolling': '#f39c12',
        'circulating': '#95a5a6'
    }
    
    colors = [state_colors.get(s, '#95a5a6') for s in states]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=states,
        hovertemplate='<b>%{text}</b><br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}'
    )])
    
    fig.update_layout(
        title="cell positions",
        scene=dict(
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            zaxis_title="z (μm)",
            aspectmode='data'
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig


def create_metrics_plot(time_points: list, mean_speeds: list) -> go.Figure:
    """create time series metrics plot"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=mean_speeds,
        mode='lines',
        name='mean speed',
        line=dict(color='#3498db', width=2)
    ))
    
    fig.update_layout(
        title="population metrics",
        xaxis_title="time (s)",
        yaxis_title="mean speed (μm/s)",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified'
    )
    
    return fig


def display_analysis(results: dict):
    """display detailed analysis"""
    
    st.subheader("trajectory analysis")
    
    # trajectory plot
    positions = results['positions']
    
    if len(positions) > 0:
        # select random cells to plot
        n_cells = len(positions[0])
        n_plot = min(10, n_cells)
        cell_indices = np.random.choice(n_cells, n_plot, replace=False)
        
        fig = go.Figure()
        
        for idx in cell_indices:
            traj = np.array([pos[idx] for pos in positions])
            
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode='lines+markers',
                marker=dict(size=3),
                line=dict(width=2),
                name=f'cell {idx}'
            ))
        
        fig.update_layout(
            title=f"trajectories ({n_plot} cells)",
            scene=dict(
                xaxis_title="x (μm)",
                yaxis_title="y (μm)",
                zaxis_title="z (μm)"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # displacement distribution
    st.subheader("displacement distribution")
    
    initial_pos = positions[0]
    final_pos = positions[-1]
    displacements = np.linalg.norm(final_pos - initial_pos, axis=1)
    
    fig = go.Figure(data=[go.Histogram(
        x=displacements,
        nbinsx=30,
        marker_color='#3498db'
    )])
    
    fig.update_layout(
        title="total displacement",
        xaxis_title="displacement (μm)",
        yaxis_title="count",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # speed over time
    st.subheader("migration speed")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['mean_speeds'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig.update_layout(
        title="mean population speed",
        xaxis_title="time (s)",
        yaxis_title="speed (μm/s)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_receptor_info():
    """display receptor information"""
    
    st.subheader("receptor library")
    
    # receptor table
    receptor_data = []
    for name, props in RECEPTOR_LIBRARY.items():
        receptor_data.append({
            'receptor': name,
            'family': props.family.value,
            'kd (nm)': f"{props.kd_nm:.1f}",
            'density': props.density,
            'kon (m⁻¹s⁻¹)': f"{props.kon:.1e}",
            'koff (s⁻¹)': f"{props.koff:.3f}"
        })
    
    df_receptors = pd.DataFrame(receptor_data)
    st.dataframe(df_receptors, use_container_width=True)
    
    st.markdown("---")
    
    # chemokine table
    st.subheader("chemokine library")
    
    chemokine_data = []
    for name, props in CHEMOKINE_LIBRARY.items():
        chemokine_data.append({
            'chemokine': props.name,
            'family': props.family,
            'mw (kda)': props.molecular_weight,
            'diffusion (μm²/s)': props.diffusion_coeff,
            'targets': ', '.join(props.receptor_targets)
        })
    
    df_chemokines = pd.DataFrame(chemokine_data)
    st.dataframe(df_chemokines, use_container_width=True)
    
    st.markdown("---")
    
    # binding kinetics visualization
    st.subheader("binding kinetics")
    
    selected_receptor = st.selectbox(
        "select receptor",
        list(RECEPTOR_LIBRARY.keys())
    )
    
    props = RECEPTOR_LIBRARY[selected_receptor]
    
    # simulate binding curve
    concentrations = np.logspace(-11, -6, 100)  # 0.1 nM to 1 μM
    occupancy = concentrations / (props.kd + concentrations)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=concentrations * 1e9,  # convert to nM
        y=occupancy * 100,
        mode='lines',
        line=dict(color='#9b59b6', width=3)
    ))
    
    # mark kd
    fig.add_vline(x=props.kd_nm, line_dash="dash", line_color="red",
                  annotation_text=f"kd = {props.kd_nm:.1f} nm")
    
    fig.update_layout(
        title=f"{selected_receptor} binding curve",
        xaxis_title="ligand concentration (nm)",
        yaxis_title="receptor occupancy (%)",
        xaxis_type="log",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_documentation():
    """display documentation"""
    
    st.subheader("model documentation")
    
    st.markdown("""
    ### mathematical framework
    
    #### receptor binding kinetics
    """)
    
    st.latex(r"\frac{d[RL]}{dt} = k_{on}[R][L] - k_{off}[RL]")
    
    st.markdown("""
    - equilibrium dissociation constant: $K_d = \\frac{k_{off}}{k_{on}}$
    - occupancy at equilibrium: $\\theta = \\frac{[L]}{K_d + [L]}$
    """)
    
    st.markdown("""
    #### chemokine diffusion
    """)
    
    st.latex(r"\frac{\partial C}{\partial t} = D\nabla^2 C - kC + S")
    
    st.markdown("""
    - $D$: diffusion coefficient (μm²/s)
    - $k$: degradation rate (s⁻¹)
    - $S$: production term (molecules/μm³/s)
    """)
    
    st.markdown("""
    #### signal transduction
    
    ordinary differential equations for pathway components:
    """)
    
    st.latex(r"\frac{d[X]}{dt} = k_{act}[Input] - k_{deact}[X]")
    
    st.markdown("""
    #### cell migration
    
    persistent random walk with chemotactic bias:
    """)
    
    st.latex(r"\vec{v}(t+dt) = \vec{v}(t) + \beta \nabla C + \eta(t)")
    
    st.markdown("""
    - $\\beta$: chemotactic sensitivity
    - $\\nabla C$: concentration gradient
    - $\\eta(t)$: random perturbation
    
    ### references
    
    1. **receptor signaling**: pierce et al., nat rev immunol (2002)
    2. **chemotaxis**: iglesias & devreotes, curr opin cell biol (2012)
    3. **leukocyte trafficking**: ley et al., nat rev immunol (2007)
    4. **gradient sensing**: Parent & devreotes, science (1999)
    5. **migration dynamics**: friedl & weigelin, nat immunol (2008)
    
    ### parameter sources
    
    - receptor affinities: chemokine receptor binding studies
    - diffusion coefficients: fluorescence recovery after photobleaching (frap)
    - migration speeds: intravital microscopy measurements
    - signaling kinetics: biochemical assays and modeling studies
    """)


if __name__ == "__main__":
    main()
