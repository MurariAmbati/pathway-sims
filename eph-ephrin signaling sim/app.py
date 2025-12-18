"""
Streamlit App for Eph/Ephrin Signaling Simulation

Interactive visualization of Eph/Ephrin signaling dynamics, tissue boundary formation,
and axon guidance mechanisms.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from eph_ephrin_model import (
    EphEphrinSignaling, 
    EphEphrinParameters,
    analyze_gradient_response,
    simulate_ephrin_pulse
)
from spatial_model import (
    TissueBoundaryModel,
    SpatialParameters,
    CellPopulation
)
from axon_guidance import (
    AxonGuidanceSimulation,
    AxonGuidanceParameters,
    EphrinGradient,
    TopographicMapping
)


st.set_page_config(page_title="eph/ephrin signaling", layout="wide")


def main():
    st.title("eph/ephrin signaling simulation")
    
    st.markdown("""
    comprehensive model of eph receptor and ephrin ligand signaling in neural development.
    explore bidirectional signaling, tissue boundary formation, and axon guidance dynamics.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("simulation modules")
    module = st.sidebar.radio(
        "select module:",
        ["molecular signaling", "tissue boundaries", "axon guidance", "topographic mapping"]
    )
    
    if module == "molecular signaling":
        molecular_signaling_module()
    elif module == "tissue boundaries":
        tissue_boundary_module()
    elif module == "axon guidance":
        axon_guidance_module()
    elif module == "topographic mapping":
        topographic_mapping_module()


def molecular_signaling_module():
    """Module for molecular-level signaling dynamics"""
    st.header("molecular signaling dynamics")
    
    st.markdown("""
    bidirectional signaling cascade with forward (into eph cell) and 
    reverse (into ephrin cell) pathways. includes receptor clustering,
    activation, and downstream rho/rac gtpase regulation.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("parameters")
        
        # Binding parameters
        st.write("**binding kinetics**")
        k_bind = st.slider("binding rate", 0.1, 5.0, 1.0, 0.1)
        k_unbind = st.slider("unbinding rate", 0.01, 1.0, 0.1, 0.01)
        
        # Clustering
        st.write("**receptor clustering**")
        k_cluster = st.slider("clustering rate", 0.1, 2.0, 0.5, 0.1)
        k_activation = st.slider("activation rate", 0.5, 5.0, 2.0, 0.1)
        
        # Downstream signaling
        st.write("**downstream effectors**")
        k_rac = st.slider("rac activation", 0.5, 3.0, 1.5, 0.1)
        k_rho = st.slider("rho activation", 0.5, 3.0, 2.0, 0.1)
        
        # External stimulus
        st.write("**external stimulus**")
        external_ephrin = st.slider("external ephrin", 0.0, 20.0, 5.0, 0.5)
        
        sim_time = st.slider("simulation time", 10, 100, 50, 5)
    
    with col2:
        # Create model with custom parameters
        params = EphEphrinParameters(
            k_bind=k_bind,
            k_unbind=k_unbind,
            k_cluster=k_cluster,
            k_activation=k_activation,
            k_rac_act=k_rac,
            k_rho_act=k_rho
        )
        
        model = EphEphrinSignaling(params)
        t, solution = model.simulate(
            (0, sim_time),
            n_points=1000,
            external_ephrin=external_ephrin
        )
        
        # Plot time courses
        st.subheader("temporal dynamics")
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("receptor dynamics", "signaling activation", "cytoskeletal regulators"),
            vertical_spacing=0.12
        )
        
        # Receptor dynamics
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 0], name="free eph", 
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 2], name="complex",
                      line=dict(color='purple')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 3], name="clusters",
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Signaling
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 4], name="active eph",
                      line=dict(color='darkred')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 5], name="forward signal",
                      line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 6], name="reverse signal",
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        # Rho/Rac
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 7], name="rac-gtp",
                      line=dict(color='cyan')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=solution[:, 8], name="rhoa-gtp",
                      line=dict(color='magenta')),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="time", row=3, col=1)
        fig.update_yaxes(title_text="concentration", row=1, col=1)
        fig.update_yaxes(title_text="activity", row=2, col=1)
        fig.update_yaxes(title_text="gtp-bound", row=3, col=1)
        
        fig.update_layout(height=900, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Phase plane analysis
        st.subheader("phase plane: rho vs rac")
        
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=solution[:, 7],  # Rac
            y=solution[:, 8],  # Rho
            mode='lines+markers',
            marker=dict(
                size=4,
                color=t,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="time")
            ),
            line=dict(color='gray', width=1),
            showlegend=False
        ))
        
        # Add nullclines or decision boundary
        fig_phase.add_shape(
            type="line",
            x0=0, y0=0, x1=max(solution[:, 7]), y1=max(solution[:, 7]),
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig_phase.update_layout(
            xaxis_title="rac-gtp (attractive)",
            yaxis_title="rhoa-gtp (repulsive)",
            height=400
        )
        st.plotly_chart(fig_phase, use_container_width=True)
        
        # Calculate net response
        net_response = solution[:, 8] - solution[:, 7]
        st.subheader("net cellular response")
        
        fig_response = go.Figure()
        fig_response.add_trace(go.Scatter(
            x=t, y=net_response,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='darkred', width=2)
        ))
        fig_response.add_hline(y=0, line_dash="dash", line_color="black")
        fig_response.update_layout(
            xaxis_title="time",
            yaxis_title="net response (rho - rac)",
            height=300
        )
        st.plotly_chart(fig_response, use_container_width=True)
        
        final_response = net_response[-1]
        if final_response > 0.5:
            st.success(f"**repulsion** (net response: {final_response:.2f})")
        elif final_response < -0.5:
            st.info(f"**attraction** (net response: {final_response:.2f})")
        else:
            st.warning(f"**neutral** (net response: {final_response:.2f})")


def tissue_boundary_module():
    """Module for tissue boundary formation through cell sorting"""
    st.header("tissue boundary formation")
    
    st.markdown("""
    spatial simulation of cell sorting and boundary sharpening through
    differential eph/ephrin expression and adhesion.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("setup")
        
        # Cell populations
        st.write("**cell populations**")
        n_populations = st.radio("number of populations", [2, 3], 0)
        
        populations = []
        colors = ['red', 'blue', 'green', 'yellow']
        
        for i in range(n_populations):
            st.write(f"population {i+1}")
            eph = st.slider(f"eph level (pop {i+1})", 0.0, 10.0, 
                          float(5.0 * (i + 1)), 0.5, key=f"eph_{i}")
            ephrin = st.slider(f"ephrin level (pop {i+1})", 0.0, 10.0,
                             float(5.0 * (n_populations - i)), 0.5, key=f"ephrin_{i}")
            
            populations.append(CellPopulation(
                name=f"population_{i+1}",
                eph_level=eph,
                ephrin_level=ephrin,
                color=colors[i]
            ))
        
        st.write("**spatial parameters**")
        repulsion = st.slider("repulsion strength", 0.0, 5.0, 2.0, 0.1)
        homotypic = st.slider("homotypic adhesion", 0.0, 2.0, 1.0, 0.1)
        heterotypic = st.slider("heterotypic adhesion", 0.0, 2.0, 0.3, 0.1)
        
        layout = st.selectbox("initial layout", 
                             ["side_by_side", "mixed", "concentric"])
        
        n_steps = st.slider("simulation steps", 50, 500, 200, 50)
        
        if st.button("run simulation", type="primary"):
            st.session_state.run_boundary_sim = True
            st.session_state.boundary_params = {
                'populations': populations,
                'repulsion': repulsion,
                'homotypic': homotypic,
                'heterotypic': heterotypic,
                'layout': layout,
                'n_steps': n_steps
            }
    
    with col2:
        if 'run_boundary_sim' in st.session_state and st.session_state.run_boundary_sim:
            params_dict = st.session_state.boundary_params
            
            # Create model
            params = SpatialParameters(
                grid_size=(80, 80),
                repulsion_strength=params_dict['repulsion'],
                homotypic_adhesion=params_dict['homotypic'],
                heterotypic_adhesion=params_dict['heterotypic']
            )
            
            model = TissueBoundaryModel(params)
            model.initialize_populations(
                params_dict['populations'],
                layout=params_dict['layout']
            )
            
            # Record initial state
            initial_grid = model.grid.copy()
            initial_segregation = model.measure_segregation_index()
            
            # Run simulation
            with st.spinner("simulating cell sorting..."):
                snapshots = model.simulate(
                    params_dict['n_steps'],
                    record_interval=max(1, params_dict['n_steps'] // 10)
                )
            
            final_segregation = model.measure_segregation_index()
            
            st.subheader("boundary formation dynamics")
            
            # Show initial and final states
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Initial state
            axes[0].imshow(initial_grid, cmap='tab10', interpolation='nearest')
            axes[0].set_title(f"initial state\nsegregation: {initial_segregation:.3f}")
            axes[0].axis('off')
            
            # Final state
            axes[1].imshow(model.grid, cmap='tab10', interpolation='nearest')
            axes[1].set_title(f"final state\nsegregation: {final_segregation:.3f}")
            axes[1].axis('off')
            
            # Boundary detection
            boundary_mask = model.get_boundary_cells()
            axes[2].imshow(model.grid, cmap='tab10', alpha=0.5, interpolation='nearest')
            axes[2].imshow(boundary_mask, cmap='hot', alpha=0.5, interpolation='nearest')
            axes[2].set_title("detected boundaries")
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Metrics over time
            st.subheader("quantitative analysis")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("initial segregation", f"{initial_segregation:.3f}")
                st.metric("final segregation", f"{final_segregation:.3f}")
                st.metric("segregation increase", 
                         f"{(final_segregation - initial_segregation):.3f}")
            
            with col_b:
                boundary_sharpness = model.measure_boundary_sharpness()
                st.metric("boundary sharpness", f"{boundary_sharpness:.3f}")
                
                # Energy
                energy = model.calculate_adhesion_energy()
                st.metric("adhesion energy", f"{energy:.1f}")
            
            # Show ephrin fields
            st.subheader("molecular expression patterns")
            
            fig_fields, axes_fields = plt.subplots(1, 3, figsize=(15, 4))
            
            im1 = axes_fields[0].imshow(model.eph_field, cmap='Reds', interpolation='bilinear')
            axes_fields[0].set_title("eph receptor")
            axes_fields[0].axis('off')
            plt.colorbar(im1, ax=axes_fields[0])
            
            im2 = axes_fields[1].imshow(model.ephrin_field, cmap='Blues', interpolation='bilinear')
            axes_fields[1].set_title("ephrin ligand")
            axes_fields[1].axis('off')
            plt.colorbar(im2, ax=axes_fields[1])
            
            repulsion_field = model.calculate_repulsion_field()
            im3 = axes_fields[2].imshow(repulsion_field, cmap='RdYlBu_r', interpolation='bilinear')
            axes_fields[2].set_title("repulsion field")
            axes_fields[2].axis('off')
            plt.colorbar(im3, ax=axes_fields[2])
            
            plt.tight_layout()
            st.pyplot(fig_fields)
            plt.close()


def axon_guidance_module():
    """Module for axon guidance simulation"""
    st.header("axon guidance simulation")
    
    st.markdown("""
    growth cone navigation through ephrin gradients with chemotactic
    and chemorepulsive responses.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("parameters")
        
        # Gradient type
        st.write("**ephrin gradient**")
        gradient_type = st.selectbox(
            "pattern",
            ["linear", "exponential", "striped", "circular barrier"]
        )
        
        if gradient_type == "linear":
            direction = st.radio("direction", ["horizontal", "vertical"])
            max_ephrin = st.slider("max ephrin", 1.0, 20.0, 10.0, 1.0)
        elif gradient_type == "exponential":
            source_x = st.slider("source x", 0, 100, 80, 5)
            source_y = st.slider("source y", 0, 100, 50, 5)
            decay = st.slider("decay length", 10.0, 50.0, 20.0, 5.0)
        elif gradient_type == "striped":
            stripe_width = st.slider("stripe width", 5, 20, 10, 1)
        
        st.write("**growth cone**")
        n_cones = st.slider("number of growth cones", 1, 10, 3, 1)
        growth_speed = st.slider("growth speed", 0.1, 2.0, 1.0, 0.1)
        sensitivity = st.slider("turning sensitivity", 0.1, 2.0, 0.5, 0.1)
        persistence = st.slider("persistence length", 5.0, 30.0, 10.0, 1.0)
        
        if st.button("run guidance simulation", type="primary"):
            st.session_state.run_guidance = True
            st.session_state.guidance_params = {
                'gradient_type': gradient_type,
                'n_cones': n_cones,
                'growth_speed': growth_speed,
                'sensitivity': sensitivity,
                'persistence': persistence
            }
            
            if gradient_type == "linear":
                st.session_state.guidance_params['direction'] = direction
                st.session_state.guidance_params['max_ephrin'] = max_ephrin
            elif gradient_type == "exponential":
                st.session_state.guidance_params['source_x'] = source_x
                st.session_state.guidance_params['source_y'] = source_y
                st.session_state.guidance_params['decay'] = decay
            elif gradient_type == "striped":
                st.session_state.guidance_params['stripe_width'] = stripe_width
    
    with col2:
        if 'run_guidance' in st.session_state and st.session_state.run_guidance:
            params_dict = st.session_state.guidance_params
            
            # Create gradient field
            grid_size = (100, 100)
            
            if params_dict['gradient_type'] == "linear":
                ephrin_field = EphrinGradient.linear_gradient(
                    grid_size,
                    direction=params_dict['direction'],
                    max_val=params_dict['max_ephrin']
                )
            elif params_dict['gradient_type'] == "exponential":
                ephrin_field = EphrinGradient.exponential_gradient(
                    grid_size,
                    source_position=(params_dict['source_y'], params_dict['source_x']),
                    decay_length=params_dict['decay']
                )
            elif params_dict['gradient_type'] == "striped":
                ephrin_field = EphrinGradient.striped_pattern(
                    grid_size,
                    stripe_width=params_dict['stripe_width']
                )
            elif params_dict['gradient_type'] == "circular barrier":
                ephrin_field = EphrinGradient.circular_barrier(
                    grid_size,
                    center=(50, 50)
                )
            
            # Create simulation
            guidance_params = AxonGuidanceParameters(
                growth_speed=params_dict['growth_speed'],
                turning_sensitivity=params_dict['sensitivity'],
                persistence_length=params_dict['persistence']
            )
            
            sim = AxonGuidanceSimulation(ephrin_field, guidance_params)
            
            # Add growth cones from left side
            for i in range(params_dict['n_cones']):
                start_y = int((i + 1) * grid_size[0] / (params_dict['n_cones'] + 1))
                sim.add_growth_cone((5, start_y), direction=0.0)
            
            # Run simulation
            with st.spinner("simulating axon guidance..."):
                trajectories = sim.simulate(max_iterations=400)
            
            st.subheader("growth cone trajectories")
            
            # Plot trajectories
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Show ephrin field
            im = ax.imshow(ephrin_field, cmap='YlOrRd', alpha=0.6, 
                          extent=[0, grid_size[1], 0, grid_size[0]],
                          origin='lower')
            plt.colorbar(im, ax=ax, label='ephrin concentration')
            
            # Plot trajectories
            colors_traj = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
            
            for idx, traj in enumerate(trajectories):
                traj_array = np.array(traj)
                if len(traj_array) > 0:
                    ax.plot(traj_array[:, 0], traj_array[:, 1],
                           color=colors_traj[idx], linewidth=2, alpha=0.8,
                           label=f'axon {idx+1}')
                    
                    # Mark start and end
                    ax.plot(traj_array[0, 0], traj_array[0, 1], 'go', 
                           markersize=10, markeredgecolor='black')
                    ax.plot(traj_array[-1, 0], traj_array[-1, 1], 'rs',
                           markersize=10, markeredgecolor='black')
            
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
            ax.set_title('axon guidance through ephrin gradient')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim(0, grid_size[1])
            ax.set_ylim(0, grid_size[0])
            
            st.pyplot(fig)
            plt.close()
            
            # Analysis
            st.subheader("trajectory analysis")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                avg_length = np.mean([len(t) for t in trajectories])
                st.metric("average path length", f"{avg_length:.1f} steps")
            
            with col_b:
                # Calculate tortuosity (path length / straight-line distance)
                tortuosities = []
                for traj in trajectories:
                    if len(traj) > 1:
                        traj_arr = np.array(traj)
                        path_length = np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1))
                        straight = np.linalg.norm(traj_arr[-1] - traj_arr[0])
                        if straight > 0:
                            tortuosities.append(path_length / straight)
                
                if tortuosities:
                    avg_tort = np.mean(tortuosities)
                    st.metric("average tortuosity", f"{avg_tort:.2f}")
            
            with col_c:
                # Final positions
                final_x = np.mean([t[-1][0] for t in trajectories if len(t) > 0])
                st.metric("mean final x", f"{final_x:.1f}")


def topographic_mapping_module():
    """Module for topographic map formation"""
    st.header("topographic mapping")
    
    st.markdown("""
    retinotectal map formation through complementary eph/ephrin gradients.
    axons from retina project to tectum in ordered topographic fashion.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("parameters")
        
        n_axons = st.slider("number of axons", 5, 30, 15, 5)
        
        st.write("**guidance parameters**")
        sensitivity = st.slider("gradient sensitivity", 0.1, 2.0, 0.5, 0.1)
        speed = st.slider("growth speed", 0.3, 2.0, 1.0, 0.1)
        
        if st.button("generate topographic map", type="primary"):
            st.session_state.run_topo = True
            st.session_state.topo_params = {
                'n_axons': n_axons,
                'sensitivity': sensitivity,
                'speed': speed
            }
    
    with col2:
        if 'run_topo' in st.session_state and st.session_state.run_topo:
            params_dict = st.session_state.topo_params
            
            # Create topographic mapping
            topo = TopographicMapping(source_size=50, target_size=100)
            
            guidance_params = AxonGuidanceParameters(
                growth_speed=params_dict['speed'],
                turning_sensitivity=params_dict['sensitivity'],
                persistence_length=15.0
            )
            
            with st.spinner("generating topographic map..."):
                trajectories = topo.simulate_topographic_map(
                    n_axons=params_dict['n_axons'],
                    params=guidance_params
                )
            
            st.subheader("retinotectal projection map")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Left: Ephrin gradient in tectum
            im1 = ax1.imshow(topo.target_ephrin_a, cmap='RdYlGn_r', 
                           extent=[0, 100, 0, 100], origin='lower')
            
            # Plot trajectories
            colors_map = plt.cm.plasma(np.linspace(0, 1, len(trajectories)))
            
            for idx, traj in enumerate(trajectories):
                traj_array = np.array(traj)
                if len(traj_array) > 0:
                    ax1.plot(traj_array[:, 0], traj_array[:, 1],
                           color=colors_map[idx], linewidth=2, alpha=0.7)
                    ax1.plot(traj_array[-1, 0], traj_array[-1, 1], 'o',
                           color=colors_map[idx], markersize=8,
                           markeredgecolor='black', markeredgewidth=1)
            
            ax1.set_xlabel('anterior - posterior')
            ax1.set_ylabel('medial - lateral')
            ax1.set_title('axon projections on ephrina gradient')
            plt.colorbar(im1, ax=ax1, label='ephrina level')
            
            # Right: Topographic organization
            # Extract final positions
            source_positions = []
            target_positions = []
            
            for idx, traj in enumerate(trajectories):
                if len(traj) > 1:
                    source_y = idx * 50 / len(trajectories)
                    source_positions.append(source_y)
                    
                    traj_array = np.array(traj)
                    target_positions.append(traj_array[-1, 1])
            
            if source_positions:
                ax2.scatter(source_positions, target_positions, 
                          c=colors_map[:len(source_positions)], 
                          s=100, edgecolors='black', linewidths=2)
                
                # Ideal linear mapping
                ideal_x = np.linspace(0, 50, 100)
                ideal_y = ideal_x * 2  # Map 0-50 to 0-100
                ax2.plot(ideal_x, ideal_y, 'k--', linewidth=2, 
                        alpha=0.5, label='ideal mapping')
            
            ax2.set_xlabel('retinal position (anterior - posterior)')
            ax2.set_ylabel('tectal position (anterior - posterior)')
            ax2.set_title('topographic organization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Calculate mapping quality
            if source_positions:
                st.subheader("mapping quality")
                
                # Correlation between source and target positions
                correlation = np.corrcoef(source_positions, target_positions)[0, 1]
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("topographic correlation", f"{correlation:.3f}")
                
                with col_b:
                    # Calculate mapping precision (std of residuals from ideal)
                    ideal_targets = np.array(source_positions) * 2
                    residuals = np.array(target_positions) - ideal_targets
                    precision = np.std(residuals)
                    st.metric("mapping precision (std)", f"{precision:.2f}")
                
                if correlation > 0.8:
                    st.success("strong topographic organization achieved")
                elif correlation > 0.5:
                    st.warning("moderate topographic organization")
                else:
                    st.error("weak topographic organization")


if __name__ == "__main__":
    main()
