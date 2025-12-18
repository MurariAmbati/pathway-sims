"""
Streamlit app for NLRP3 Inflammasome Activation Simulation
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from nlrp3_model import (
    NLRP3Model, Parameters, 
    step_signal, pulse_signal, ramp_signal, oscillating_signal
)


# Page configuration
st.set_page_config(
    page_title="NLRP3 Inflammasome Activation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_signal_profile(signal_type, params, t_max):
    """Create signal profile based on user selection"""
    if signal_type == "Step":
        return step_signal(params['t_start'], params['amplitude'])
    elif signal_type == "Pulse":
        return pulse_signal(params['t_start'], params['duration'], params['amplitude'])
    elif signal_type == "Ramp":
        return ramp_signal(params['t_start'], params['t_end'], params['amplitude'])
    elif signal_type == "Oscillating":
        return oscillating_signal(params['period'], params['amplitude'], params['phase'])
    else:  # None
        return lambda t: 0.0


def plot_pathway_dynamics(t, states_dict, selected_components):
    """Create multi-panel plot of pathway dynamics"""
    n_plots = len(selected_components)
    fig = make_subplots(
        rows=n_plots, cols=1,
        subplot_titles=selected_components,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, component in enumerate(selected_components, 1):
        if component in states_dict:
            fig.add_trace(
                go.Scatter(
                    x=t, y=states_dict[component],
                    name=component,
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    showlegend=False
                ),
                row=i, col=1
            )
            fig.update_yaxes(title_text="Conc. (AU)", row=i, col=1)
    
    fig.update_xaxes(title_text="Time (min)", row=n_plots, col=1)
    fig.update_layout(
        height=250 * n_plots,
        template="plotly_white",
        font=dict(size=11)
    )
    
    return fig


def plot_heatmap(t, states_dict, components):
    """Create heatmap of component dynamics"""
    data = np.array([states_dict[comp] for comp in components if comp in states_dict])
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=t,
        y=[comp for comp in components if comp in states_dict],
        colorscale='RdYlBu_r',
        colorbar=dict(title="Concentration")
    ))
    
    fig.update_layout(
        title="Pathway Component Heatmap",
        xaxis_title="Time (min)",
        yaxis_title="Component",
        height=400,
        template="plotly_white"
    )
    
    return fig


def plot_phase_space(states_dict, x_var, y_var):
    """Create phase space plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=states_dict[x_var],
        y=states_dict[y_var],
        mode='lines+markers',
        marker=dict(
            size=4,
            color=np.linspace(0, 1, len(states_dict[x_var])),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time")
        ),
        line=dict(width=1.5),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Phase Space: {x_var} vs {y_var}",
        xaxis_title=x_var,
        yaxis_title=y_var,
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_outcomes(t, states_dict):
    """Plot key outcomes: cytokines and pyroptosis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Mature Cytokines",
            "Cell Viability",
            "GSDMD Cleavage",
            "Caspase-1 Activity"
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cytokines
    fig.add_trace(
        go.Scatter(x=t, y=states_dict['IL-1β (mature)'], name='IL-1β', 
                   line=dict(color='red', width=2.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=states_dict['IL-18 (mature)'], name='IL-18',
                   line=dict(color='orange', width=2.5)),
        row=1, col=1
    )
    
    # Viability
    fig.add_trace(
        go.Scatter(x=t, y=states_dict['Cell viability'], name='Viability',
                   line=dict(color='green', width=2.5), showlegend=False),
        row=1, col=2
    )
    
    # GSDMD
    fig.add_trace(
        go.Scatter(x=t, y=states_dict['GSDMD'], name='GSDMD (intact)',
                   line=dict(color='blue', width=2.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=states_dict['GSDMD-NT'], name='GSDMD-NT',
                   line=dict(color='purple', width=2.5)),
        row=2, col=1
    )
    
    # Caspase-1
    fig.add_trace(
        go.Scatter(x=t, y=states_dict['Caspase-1 (active)'], name='Caspase-1',
                   line=dict(color='darkred', width=2.5), showlegend=False),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Time (min)")
    fig.update_yaxes(title_text="Concentration (AU)")
    
    fig.update_layout(
        height=600,
        template="plotly_white",
        showlegend=True
    )
    
    return fig


def plot_pathway_schematic(states_dict, t_idx):
    """Create simplified pathway schematic with current state"""
    fig = go.Figure()
    
    # Define node positions (x, y)
    nodes = {
        'Signal 1': (0, 3),
        'NF-κB': (1.5, 3),
        'Pro-IL-1β/18': (3, 3.5),
        'NLRP3': (3, 2.5),
        'Signal 2': (3, 1),
        'NLRP3*': (4.5, 2),
        'Inflammasome': (6, 2.5),
        'Caspase-1': (7.5, 2.5),
        'IL-1β/18': (9, 3.5),
        'GSDMD-NT': (9, 1.5),
        'Pyroptosis': (10.5, 1.5)
    }
    
    # Get state values at current time
    values = {
        'NF-κB': states_dict['NF-κB (active)'][t_idx],
        'NLRP3*': states_dict['NLRP3* (active)'][t_idx],
        'Inflammasome': states_dict['Inflammasome'][t_idx],
        'Caspase-1': states_dict['Caspase-1 (active)'][t_idx],
        'IL-1β/18': (states_dict['IL-1β (mature)'][t_idx] + 
                     states_dict['IL-18 (mature)'][t_idx]) / 2,
        'GSDMD-NT': states_dict['GSDMD-NT'][t_idx],
    }
    
    # Draw edges
    edges = [
        ('Signal 1', 'NF-κB'),
        ('NF-κB', 'Pro-IL-1β/18'),
        ('NF-κB', 'NLRP3'),
        ('Signal 2', 'NLRP3*'),
        ('NLRP3', 'NLRP3*'),
        ('NLRP3*', 'Inflammasome'),
        ('Inflammasome', 'Caspase-1'),
        ('Caspase-1', 'IL-1β/18'),
        ('Pro-IL-1β/18', 'IL-1β/18'),
        ('Caspase-1', 'GSDMD-NT'),
        ('GSDMD-NT', 'Pyroptosis')
    ]
    
    for start, end in edges:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='lightgray', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw nodes
    for name, (x, y) in nodes.items():
        size = 25
        color = 'lightgray'
        
        if name in values:
            size = 25 + values[name] * 35
            color_intensity = min(values[name], 1.0)
            r = int(100 + 155 * color_intensity)
            g = int(150 * (1 - color_intensity))
            b = int(150 * (1 - color_intensity))
            color = f'rgba({r}, {g}, {b}, 0.85)'
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(color='darkgray', width=2)),
            text=name,
            textposition='bottom center',
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hovertext=f"{name}: {values.get(name, 0):.3f}" if name in values else name,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title="NLRP3 Inflammasome Pathway Schematic",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 11.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 4]),
        height=500,
        template="plotly_white",
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    return fig


# Main app
def main():
    st.title("NLRP3 Inflammasome Activation")
    st.markdown("*Interactive simulation of IL-1β/IL-18 maturation and pyroptosis*")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Time settings
    st.sidebar.subheader("Time Course")
    t_max = st.sidebar.slider("Duration (min)", 10, 300, 120, 10)
    
    # Signal 1 (Priming)
    st.sidebar.subheader("Signal 1: Priming")
    signal1_type = st.sidebar.selectbox(
        "Signal type",
        ["None", "Step", "Pulse", "Ramp", "Oscillating"],
        index=1,
        key="s1_type"
    )
    
    signal1_params = {}
    if signal1_type != "None":
        signal1_params['amplitude'] = st.sidebar.slider(
            "Amplitude", 0.0, 1.0, 0.8, 0.1, key="s1_amp"
        )
        signal1_params['t_start'] = st.sidebar.slider(
            "Start time (min)", 0, t_max//2, 5, 1, key="s1_start"
        )
        
        if signal1_type == "Pulse":
            signal1_params['duration'] = st.sidebar.slider(
                "Duration (min)", 5, t_max, 30, 5, key="s1_dur"
            )
        elif signal1_type == "Ramp":
            signal1_params['t_end'] = st.sidebar.slider(
                "End time (min)", signal1_params['t_start'], t_max, 30, 5, key="s1_end"
            )
        elif signal1_type == "Oscillating":
            signal1_params['period'] = st.sidebar.slider(
                "Period (min)", 10, 100, 40, 5, key="s1_period"
            )
            signal1_params['phase'] = st.sidebar.slider(
                "Phase", 0.0, 2*np.pi, 0.0, 0.1, key="s1_phase"
            )
    
    # Signal 2 (Activation)
    st.sidebar.subheader("Signal 2: Activation")
    signal2_type = st.sidebar.selectbox(
        "Signal type",
        ["None", "Step", "Pulse", "Ramp", "Oscillating"],
        index=1,
        key="s2_type"
    )
    
    signal2_params = {}
    if signal2_type != "None":
        signal2_params['amplitude'] = st.sidebar.slider(
            "Amplitude", 0.0, 1.0, 0.9, 0.1, key="s2_amp"
        )
        signal2_params['t_start'] = st.sidebar.slider(
            "Start time (min)", 0, t_max//2, 20, 1, key="s2_start"
        )
        
        if signal2_type == "Pulse":
            signal2_params['duration'] = st.sidebar.slider(
                "Duration (min)", 5, t_max, 60, 5, key="s2_dur"
            )
        elif signal2_type == "Ramp":
            signal2_params['t_end'] = st.sidebar.slider(
                "End time (min)", signal2_params['t_start'], t_max, 50, 5, key="s2_end"
            )
        elif signal2_type == "Oscillating":
            signal2_params['period'] = st.sidebar.slider(
                "Period (min)", 10, 100, 30, 5, key="s2_period"
            )
            signal2_params['phase'] = st.sidebar.slider(
                "Phase", 0.0, 2*np.pi, 0.0, 0.1, key="s2_phase"
            )
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        params = Parameters()
        params.k_il1b_mat = st.slider("IL-1β maturation rate", 0.5, 4.0, 2.0, 0.1)
        params.k_il18_mat = st.slider("IL-18 maturation rate", 0.5, 4.0, 1.8, 0.1)
        params.k_casp1_act = st.slider("Caspase-1 activation", 0.5, 3.0, 1.5, 0.1)
        params.k_pyroptosis = st.slider("Pyroptosis rate", 0.1, 2.0, 0.8, 0.1)
    
    # Run simulation
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Simulating pathway dynamics..."):
            # Create model and signals
            model = NLRP3Model(params)
            signal1 = create_signal_profile(signal1_type, signal1_params, t_max)
            signal2 = create_signal_profile(signal2_type, signal2_params, t_max)
            
            # Simulate
            t, states = model.simulate(
                (0, t_max),
                signal1,
                signal2,
                n_points=1000
            )
            
            states_dict = model.get_state_dict(states)
            
            # Store in session state
            st.session_state.t = t
            st.session_state.states_dict = states_dict
            st.session_state.model = model
    
    # Display results
    if 't' in st.session_state:
        t = st.session_state.t
        states_dict = st.session_state.states_dict
        model = st.session_state.model
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Outcomes", "Dynamics", "Heatmap", "Phase Space", "Pathway"
        ])
        
        with tab1:
            st.plotly_chart(plot_outcomes(t, states_dict), use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak IL-1β", f"{states_dict['IL-1β (mature)'].max():.2f}")
            with col2:
                st.metric("Peak IL-18", f"{states_dict['IL-18 (mature)'].max():.2f}")
            with col3:
                st.metric("Final Viability", f"{states_dict['Cell viability'][-1]:.1%}")
            with col4:
                st.metric("Peak Caspase-1", f"{states_dict['Caspase-1 (active)'].max():.2f}")
        
        with tab2:
            component_groups = {
                "Priming": ['NF-κB (active)', 'Pro-IL-1β', 'Pro-IL-18', 'NLRP3'],
                "Activation": ['NLRP3* (active)', 'Inflammasome', 'Caspase-1 (active)'],
                "Cytokines": ['IL-1β (mature)', 'IL-18 (mature)'],
                "Pyroptosis": ['GSDMD', 'GSDMD-NT', 'Membrane pores', 'Cell viability']
            }
            
            selected_group = st.selectbox(
                "Component group",
                list(component_groups.keys())
            )
            
            components = component_groups[selected_group]
            st.plotly_chart(
                plot_pathway_dynamics(t, states_dict, components),
                use_container_width=True
            )
        
        with tab3:
            heatmap_components = st.multiselect(
                "Select components for heatmap",
                model.state_names,
                default=['NF-κB (active)', 'NLRP3* (active)', 'Caspase-1 (active)',
                        'IL-1β (mature)', 'IL-18 (mature)', 'GSDMD-NT', 'Cell viability']
            )
            if heatmap_components:
                st.plotly_chart(
                    plot_heatmap(t, states_dict, heatmap_components),
                    use_container_width=True
                )
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X-axis", model.state_names, index=4)
            with col2:
                y_var = st.selectbox("Y-axis", model.state_names, index=8)
            
            st.plotly_chart(
                plot_phase_space(states_dict, x_var, y_var),
                use_container_width=True
            )
        
        with tab5:
            time_point = st.slider(
                "Time point (min)",
                0, int(t[-1]), int(t[-1]//2), 1
            )
            t_idx = np.argmin(np.abs(t - time_point))
            
            st.plotly_chart(
                plot_pathway_schematic(states_dict, t_idx),
                use_container_width=True
            )
            
            st.markdown(f"**Time: {t[t_idx]:.1f} min** | Viability: {states_dict['Cell viability'][t_idx]:.1%}")
    
    else:
        st.info("Configure parameters in sidebar and click 'Run Simulation'")


if __name__ == "__main__":
    main()
