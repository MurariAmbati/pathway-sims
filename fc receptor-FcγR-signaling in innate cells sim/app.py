"""
FcγR Signaling Simulator - Streamlit Application
Comprehensive simulation of Fc receptor signaling in innate immune cells
"""

import streamlit as st
import numpy as np
import pandas as pd
from fcgr_models import (
    FcgRSignalingModel, 
    FcgRParameters,
    AntibodyDependentPhagocytosis,
    AntibodyDependentCytotoxicity,
    FcgRCrosslinking
)
from visualization import SignalingVisualizer, ComparisonVisualizer


# Page configuration
st.set_page_config(
    page_title="FcγR Signaling Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">FcγR Signaling Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Antibody-Dependent Phagocytosis and Cytotoxicity in Innate Cells</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        [
            "Overview",
            "Signaling Cascade",
            "Phagocytosis (ADCP)",
            "Cytotoxicity (ADCC)",
            "FcγR Clustering",
            "Parameter Sensitivity",
            "Cell Type Comparison"
        ]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Signaling Cascade":
        show_signaling_cascade()
    elif page == "Phagocytosis (ADCP)":
        show_phagocytosis()
    elif page == "Cytotoxicity (ADCC)":
        show_cytotoxicity()
    elif page == "FcγR Clustering":
        show_clustering()
    elif page == "Parameter Sensitivity":
        show_sensitivity_analysis()
    elif page == "Cell Type Comparison":
        show_cell_comparison()


def show_overview():
    """Overview page"""
    st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Fc Receptor (FcγR) Biology
        
        **Fc receptors** are critical components of antibody-mediated immune responses. They bind the 
        Fc region of immunoglobulins and trigger various cellular responses in innate immune cells.
        
        #### Key FcγR Types in Humans:
        - **FcγRI (CD64)**: High-affinity receptor, binds monomeric IgG
        - **FcγRIIA (CD32a)**: Low-affinity activating receptor
        - **FcγRIIB (CD32b)**: Inhibitory receptor with ITIM motif
        - **FcγRIIIA (CD16a)**: Low-affinity activating receptor (NK cells)
        - **FcγRIIIB (CD16b)**: GPI-anchored receptor (neutrophils)
        
        #### Signaling Mechanisms:
        - **ITAM signaling**: Immunoreceptor tyrosine-based activation motif
        - **ITIM signaling**: Immunoreceptor tyrosine-based inhibitory motif
        - **Crosslinking**: Receptor clustering amplifies signaling
        """)
        
    with col2:
        st.markdown("""
        ### Functional Outcomes
        
        #### Antibody-Dependent Cellular Phagocytosis (ADCP)
        - Opsonized target recognition
        - Phagosome formation and maturation
        - Antigen presentation
        - Pro-inflammatory cytokine production
        
        #### Antibody-Dependent Cellular Cytotoxicity (ADCC)
        - Direct target cell killing
        - Perforin/granzyme release
        - Death receptor engagement
        - Tumor immunosurveillance
        
        #### Downstream Signaling:
        - **PI3K/Akt pathway**: Survival, metabolism
        - **MAPK cascade**: Gene transcription
        - **PLCγ/Ca²⁺**: Degranulation, cytokine release
        - **NF-κB/NFAT**: Inflammatory gene expression
        """)
    
    # Display pathway network
    st.markdown('<div class="section-header">Signaling Network Architecture</div>', unsafe_allow_html=True)
    
    include_inhibitory = st.checkbox("Include inhibitory pathways (FcγRIIB/SHIP)", value=True)
    
    visualizer = SignalingVisualizer()
    fig = visualizer.plot_pathway_network(include_inhibitory=include_inhibitory)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick facts
    st.markdown('<div class="section-header">Key Facts</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("FcγR per Macrophage", "50,000 - 200,000")
    with col2:
        st.metric("IgG-FcγR Affinity (Kd)", "10⁻⁷ - 10⁻⁸ M")
    with col3:
        st.metric("ADCP Duration", "15-60 min")
    with col4:
        st.metric("ADCC Duration", "2-6 hours")


def show_signaling_cascade():
    """Signaling cascade simulation"""
    st.markdown('<div class="section-header">FcγR Signaling Cascade</div>', unsafe_allow_html=True)
    
    # Sidebar parameters
    st.sidebar.markdown("### Simulation Parameters")
    
    ic_concentration = st.sidebar.number_input(
        "Immune Complex Concentration (nM)",
        min_value=0.1,
        max_value=1000.0,
        value=100.0,
        step=10.0
    )
    
    simulation_time = st.sidebar.slider(
        "Simulation Time (seconds)",
        min_value=10,
        max_value=600,
        value=300,
        step=10
    )
    
    include_inhibitory = st.sidebar.checkbox("Include FcγRIIB inhibition", value=False)
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        params = FcgRParameters()
        
        params.itam_phosphorylation_rate = st.number_input(
            "ITAM phosphorylation rate (s⁻¹)",
            value=params.itam_phosphorylation_rate,
            format="%.3f"
        )
        
        params.syk_activation_rate = st.number_input(
            "Syk activation rate (s⁻¹)",
            value=params.syk_activation_rate,
            format="%.3f"
        )
        
        params.pi3k_activation_rate = st.number_input(
            "PI3K activation rate (s⁻¹)",
            value=params.pi3k_activation_rate,
            format="%.3f"
        )
        
        if include_inhibitory:
            params.fcgr2b_ratio = st.slider(
                "FcγRIIB ratio",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.05
            )
    
    # Run simulation
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            model = FcgRSignalingModel(params)
            time_points = np.linspace(0, simulation_time, 1000)
            
            # Convert nM to M
            ic_conc_m = ic_concentration * 1e-9
            
            results = model.signaling_cascade(
                ic_conc_m,
                time_points,
                include_inhibitory=include_inhibitory
            )
            
            # Store in session state
            st.session_state['cascade_results'] = results
            st.session_state['cascade_time'] = time_points
            
        st.success("Simulation complete!")
    
    # Display results
    if 'cascade_results' in st.session_state:
        results = st.session_state['cascade_results']
        time_points = st.session_state['cascade_time']
        
        # Main visualization
        visualizer = SignalingVisualizer()
        fig = visualizer.plot_signaling_cascade(results, time_points)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_syk = results['active_syk'].max()
            st.metric("Peak Syk Activation", f"{max_syk:.2%}")
        
        with col2:
            max_ca = results['calcium'].max()
            st.metric("Peak Ca²⁺ (μM)", f"{max_ca:.2f}")
        
        with col3:
            final_tnf = results['tnf_alpha'][-1]
            st.metric("Final TNF-α", f"{final_tnf:.3f}")
        
        with col4:
            if include_inhibitory:
                max_ship = results['active_ship'].max()
                st.metric("Peak SHIP Activation", f"{max_ship:.2%}")
        
        # Data export
        st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
        
        df = pd.DataFrame({
            'time': time_points,
            **{k: v for k, v in results.items()}
        })
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="fcgr_signaling_cascade.csv",
            mime="text/csv"
        )


def show_phagocytosis():
    """ADCP simulation"""
    st.markdown('<div class="section-header">Antibody-Dependent Cellular Phagocytosis</div>', unsafe_allow_html=True)
    
    # Parameters
    st.sidebar.markdown("### ADCP Parameters")
    
    cell_type = st.sidebar.selectbox(
        "Phagocyte Type",
        ["macrophage", "neutrophil", "monocyte"]
    )
    
    ab_range = st.sidebar.slider(
        "Antibody Concentration Range (μg/mL)",
        min_value=0.0,
        max_value=100.0,
        value=(0.1, 50.0),
        step=0.1
    )
    
    antigen_density = st.sidebar.number_input(
        "Target Antigen Density",
        min_value=100,
        max_value=100000,
        value=10000,
        step=1000
    )
    
    fcgr_density = st.sidebar.number_input(
        "FcγR Density per Cell",
        min_value=1000,
        max_value=200000,
        value=50000,
        step=5000
    )
    
    effector_target_ratio = st.sidebar.slider(
        "Effector:Target Ratio",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )
    
    simulation_time_min = st.sidebar.slider(
        "Simulation Time (minutes)",
        min_value=10,
        max_value=120,
        value=60,
        step=5
    )
    
    # Run simulation
    if st.button("Simulate ADCP", type="primary"):
        with st.spinner("Simulating phagocytosis..."):
            model = AntibodyDependentPhagocytosis()
            
            # Dose-response
            ab_conc = np.logspace(np.log10(ab_range[0]), np.log10(ab_range[1]), 50)
            efficiency = model.phagocytosis_efficiency(
                ab_conc,
                antigen_density,
                fcgr_density,
                cell_type
            )
            
            # Kinetics at optimal concentration
            optimal_idx = np.argmax(efficiency)
            optimal_conc = ab_conc[optimal_idx]
            optimal_opsonization = efficiency[optimal_idx]
            
            time_points = np.linspace(0, simulation_time_min, 500)
            kinetics = model.phagocytosis_kinetics(
                optimal_opsonization,
                time_points,
                effector_target_ratio
            )
            
            # Store results
            st.session_state['adcp_dose_response'] = (ab_conc, efficiency)
            st.session_state['adcp_kinetics'] = (time_points, kinetics)
            st.session_state['adcp_optimal'] = (optimal_conc, optimal_opsonization)
            
        st.success("Simulation complete!")
    
    # Display results
    if 'adcp_dose_response' in st.session_state:
        ab_conc, efficiency = st.session_state['adcp_dose_response']
        time_points, kinetics = st.session_state['adcp_kinetics']
        optimal_conc, optimal_opsonization = st.session_state['adcp_optimal']
        
        # Dose-response curve
        st.markdown("### Dose-Response Curve")
        visualizer = SignalingVisualizer()
        fig_dr = visualizer.plot_dose_response(ab_conc, efficiency, "Phagocytosis Efficiency")
        st.plotly_chart(fig_dr, use_container_width=True)
        
        # Kinetics
        st.markdown("### Phagocytosis Kinetics")
        st.info(f"Simulation at optimal antibody concentration: {optimal_conc:.2f} μg/mL (Opsonization: {optimal_opsonization:.2%})")
        
        fig_kin = visualizer.plot_phagocytosis_kinetics(kinetics, time_points)
        st.plotly_chart(fig_kin, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Optimal [Ab] (μg/mL)", f"{optimal_conc:.2f}")
        
        with col2:
            final_phagocytosed = kinetics['total_phagocytosed'][-1]
            st.metric("Final Phagocytosis", f"{final_phagocytosed:.2%}")
        
        with col3:
            half_time_idx = np.argmin(np.abs(kinetics['total_phagocytosed'] - 0.5))
            t_half = time_points[half_time_idx]
            st.metric("t₁/₂ (min)", f"{t_half:.1f}")


def show_cytotoxicity():
    """ADCC simulation"""
    st.markdown('<div class="section-header">Antibody-Dependent Cellular Cytotoxicity</div>', unsafe_allow_html=True)
    
    # Parameters
    st.sidebar.markdown("### ADCC Parameters")
    
    effector_type = st.sidebar.selectbox(
        "Effector Cell Type",
        ["nk_cell", "macrophage", "neutrophil"]
    )
    
    ab_range = st.sidebar.slider(
        "Antibody Concentration Range (μg/mL)",
        min_value=0.0,
        max_value=100.0,
        value=(0.1, 50.0),
        step=0.1
    )
    
    target_density = st.sidebar.number_input(
        "Target Antigen Density",
        min_value=100,
        max_value=100000,
        value=10000,
        step=1000
    )
    
    effector_target_ratio = st.sidebar.slider(
        "Effector:Target Ratio",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )
    
    simulation_time_hr = st.sidebar.slider(
        "Simulation Time (hours)",
        min_value=1,
        max_value=24,
        value=6,
        step=1
    )
    
    # Run simulation
    if st.button("Simulate ADCC", type="primary"):
        with st.spinner("Simulating cytotoxicity..."):
            model = AntibodyDependentCytotoxicity()
            
            # Dose-response
            ab_conc = np.logspace(np.log10(ab_range[0]), np.log10(ab_range[1]), 50)
            efficiency = model.adcc_efficiency(
                ab_conc,
                target_density,
                effector_type
            )
            
            # Kinetics at optimal concentration
            optimal_idx = np.argmax(efficiency)
            optimal_conc = ab_conc[optimal_idx]
            optimal_opsonization = efficiency[optimal_idx]
            
            time_points = np.linspace(0, simulation_time_hr, 500)
            kinetics = model.cytotoxicity_kinetics(
                optimal_opsonization,
                time_points,
                effector_target_ratio,
                effector_type
            )
            
            # Store results
            st.session_state['adcc_dose_response'] = (ab_conc, efficiency)
            st.session_state['adcc_kinetics'] = (time_points, kinetics)
            st.session_state['adcc_optimal'] = (optimal_conc, optimal_opsonization)
            
        st.success("Simulation complete!")
    
    # Display results
    if 'adcc_dose_response' in st.session_state:
        ab_conc, efficiency = st.session_state['adcc_dose_response']
        time_points, kinetics = st.session_state['adcc_kinetics']
        optimal_conc, optimal_opsonization = st.session_state['adcc_optimal']
        
        # Dose-response curve
        st.markdown("### Dose-Response Curve")
        visualizer = SignalingVisualizer()
        fig_dr = visualizer.plot_dose_response(ab_conc, efficiency, "ADCC Efficiency")
        st.plotly_chart(fig_dr, use_container_width=True)
        
        # Kinetics
        st.markdown("### Cytotoxicity Kinetics")
        st.info(f"Simulation at optimal antibody concentration: {optimal_conc:.2f} μg/mL (Opsonization: {optimal_opsonization:.2%})")
        
        fig_kin = visualizer.plot_cytotoxicity_kinetics(kinetics, time_points)
        st.plotly_chart(fig_kin, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimal [Ab] (μg/mL)", f"{optimal_conc:.2f}")
        
        with col2:
            max_lysis = kinetics['specific_lysis'].max()
            st.metric("Max Specific Lysis", f"{max_lysis:.1f}%")
        
        with col3:
            final_dead = kinetics['dead_targets'][-1]
            st.metric("Final Target Death", f"{final_dead:.2%}")
        
        with col4:
            final_exhausted = kinetics['exhausted_effectors'][-1]
            st.metric("Effector Exhaustion", f"{final_exhausted:.2%}")


def show_clustering():
    """FcγR clustering simulation"""
    st.markdown('<div class="section-header">FcγR Clustering Dynamics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Receptor Clustering:</strong> FcγR crosslinking by immune complexes leads to receptor 
    clustering, which is critical for signal amplification. Larger clusters recruit more signaling 
    molecules and generate stronger cellular responses.
    </div>
    """, unsafe_allow_html=True)
    
    # Parameters
    st.sidebar.markdown("### Clustering Parameters")
    
    ic_density = st.sidebar.slider(
        "Immune Complex Density",
        min_value=0.1,
        max_value=10.0,
        value=5.0,
        step=0.1
    )
    
    simulation_time = st.sidebar.slider(
        "Simulation Time (seconds)",
        min_value=10,
        max_value=300,
        value=120,
        step=10
    )
    
    # Run simulation
    if st.button("Simulate Clustering", type="primary"):
        with st.spinner("Simulating receptor clustering..."):
            model = FcgRCrosslinking()
            time_points = np.linspace(0, simulation_time, 500)
            
            results = model.clustering_dynamics(ic_density, time_points)
            
            st.session_state['clustering_results'] = results
            st.session_state['clustering_time'] = time_points
            
        st.success("Simulation complete!")
    
    # Display results
    if 'clustering_results' in st.session_state:
        results = st.session_state['clustering_results']
        time_points = st.session_state['clustering_time']
        
        visualizer = SignalingVisualizer()
        fig = visualizer.plot_clustering_dynamics(results, time_points)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        st.markdown("### Clustering Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            final_monomers = results['monomers'][-1]
            st.metric("Final Monomers", f"{final_monomers:.2%}")
        
        with col2:
            final_large = results['large_clusters'][-1]
            st.metric("Final Large Clusters", f"{final_large:.2%}")
        
        with col3:
            avg_size = results['average_cluster_size'][-1]
            st.metric("Average Cluster Size", f"{avg_size:.2f}")


def show_sensitivity_analysis():
    """Parameter sensitivity analysis"""
    st.markdown('<div class="section-header">Parameter Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore how different parameters affect signaling outcomes.
    """)
    
    # Select analysis type
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Signaling Cascade", "ADCP Efficiency", "ADCC Efficiency"]
    )
    
    # Parameter selection
    if analysis_type == "Signaling Cascade":
        param_names = [
            "ITAM phosphorylation rate",
            "Syk activation rate",
            "PI3K activation rate",
            "MEK activation rate"
        ]
    elif analysis_type == "ADCP Efficiency":
        param_names = [
            "Antigen density",
            "FcγR density",
            "Effector:Target ratio"
        ]
    else:  # ADCC
        param_names = [
            "Target density",
            "Effector:Target ratio",
            "Antibody concentration"
        ]
    
    selected_params = st.multiselect(
        "Select parameters to vary",
        param_names,
        default=param_names[:2]
    )
    
    if len(selected_params) >= 2:
        param1 = selected_params[0]
        param2 = selected_params[1]
        
        # Define parameter ranges
        col1, col2 = st.columns(2)
        
        with col1:
            p1_min = st.number_input(f"{param1} - Min", value=0.1)
            p1_max = st.number_input(f"{param1} - Max", value=1.0)
        
        with col2:
            p2_min = st.number_input(f"{param2} - Min", value=0.1)
            p2_max = st.number_input(f"{param2} - Max", value=1.0)
        
        if st.button("Run Sensitivity Analysis", type="primary"):
            with st.spinner("Running analysis..."):
                # Create parameter grid
                p1_values = np.linspace(p1_min, p1_max, 10)
                p2_values = np.linspace(p2_min, p2_max, 10)
                
                # Placeholder for results (would implement full analysis)
                results_matrix = np.random.rand(10, 10)  # Placeholder
                
                # Create heatmap
                df = pd.DataFrame(
                    results_matrix,
                    index=[f"{v:.2f}" for v in p2_values],
                    columns=[f"{v:.2f}" for v in p1_values]
                )
                
                visualizer = ComparisonVisualizer()
                fig = visualizer.create_heatmap(
                    df,
                    title=f"Sensitivity: {param1} vs {param2}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("This is a demonstration. Full implementation would compute actual sensitivities.")


def show_cell_comparison():
    """Compare different cell types"""
    st.markdown('<div class="section-header">Cell Type Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare FcγR expression profiles and functional responses across different innate immune cell types.
    """)
    
    # FcγR expression profiles
    st.markdown("### FcγR Expression Profiles")
    
    expression_data = pd.DataFrame({
        'Cell Type': ['Macrophage', 'Neutrophil', 'Monocyte', 'NK Cell', 'Dendritic Cell'],
        'FcγRI': ['+++', '++', '+++', '0', '++'],
        'FcγRIIA': ['+++', '+++', '+++', '0', '++'],
        'FcγRIIB': ['++', '+', '++', '0', '+'],
        'FcγRIIIA': ['++', '0', '++', '+++', '+'],
        'FcγRIIIB': ['0', '+++', '0', '0', '0']
    })
    
    st.dataframe(expression_data, use_container_width=True)
    st.caption("Expression levels: 0 (none), + (low), ++ (moderate), +++ (high)")
    
    # Functional comparison
    st.markdown("### Functional Responses")
    
    function_data = pd.DataFrame({
        'Cell Type': ['Macrophage', 'Neutrophil', 'Monocyte', 'NK Cell'],
        'ADCP': ['+++', '+++', '++', '+'],
        'ADCC': ['++', '++', '+', '+++'],
        'Cytokine Production': ['+++', '++', '+++', '++'],
        'Antigen Presentation': ['+++', '+', '++', '0']
    })
    
    st.dataframe(function_data, use_container_width=True)


def show_documentation():
    """Documentation page"""
    st.markdown('<div class="section-header">Documentation</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Description",
        "Parameters",
        "Equations",
        "References"
    ])
    
    with tab1:
        st.markdown("""
        ## Model Description
        
        ### FcγR Signaling Cascade
        
        The model simulates the complete FcγR signaling cascade from receptor engagement to 
        cytokine production:
        
        1. **Receptor Binding**: Immune complexes bind to FcγR
        2. **ITAM Phosphorylation**: Src family kinases phosphorylate ITAM motifs
        3. **Syk Recruitment**: Syk kinase binds phospho-ITAM via SH2 domains
        4. **Downstream Signaling**:
           - PI3K/Akt pathway
           - MAPK cascade (MEK/ERK)
           - PLCγ/Ca²⁺ signaling
        5. **Transcription Factors**: NFAT and NF-κB activation
        6. **Cytokine Production**: TNF-α, IL-6, IL-1β
        
        ### ADCP Model
        
        Antibody-dependent cellular phagocytosis is modeled as a multi-step process:
        
        1. **Opsonization**: Antibodies coat target cells
        2. **Recognition**: FcγR recognize opsonized targets
        3. **Binding**: Effector-target conjugate formation
        4. **Engulfment**: Phagosome formation
        5. **Digestion**: Target degradation
        
        ### ADCC Model
        
        Antibody-dependent cellular cytotoxicity follows these steps:
        
        1. **Target Recognition**: Opsonized target identification
        2. **Conjugate Formation**: Effector-target binding
        3. **Cytotoxic Attack**: Perforin/granzyme release
        4. **Target Death**: Apoptosis/necrosis
        5. **Detachment**: Effector cell release
        
        ### FcγR Clustering
        
        Receptor clustering amplifies signaling through:
        
        - Increased local concentration of signaling molecules
        - Cooperative recruitment of downstream effectors
        - Exclusion of phosphatases from signaling platforms
        """)
    
    with tab2:
        st.markdown("""
        ## Parameter Descriptions
        
        ### Receptor Parameters
        - **FcγR density**: 50,000 - 200,000 receptors/cell (cell type dependent)
        - **FcγR-IgG affinity (Kd)**: 10⁻⁷ - 10⁻⁸ M
        - **FcγRIIB ratio**: 0.1 - 0.3 (varies by cell type)
        
        ### Kinetic Parameters
        - **ITAM phosphorylation**: 0.05 - 0.2 s⁻¹
        - **Syk activation**: 0.3 - 0.7 s⁻¹
        - **PI3K activation**: 0.2 - 0.5 s⁻¹
        - **MAPK activation**: 0.3 - 0.6 s⁻¹
        
        ### Calcium Signaling
        - **Basal Ca²⁺**: 0.1 μM
        - **Peak Ca²⁺**: 1-10 μM
        - **Release rate**: 1-3 s⁻¹
        - **Reuptake rate**: 1-2 s⁻¹
        
        ### Functional Outcomes
        - **ADCP efficiency**: 0-100% (dose-dependent)
        - **ADCC efficiency**: 0-100% (dose-dependent)
        - **Time scale**: Minutes (ADCP) to hours (ADCC)
        """)
    
    with tab3:
        st.markdown("""
        ## Mathematical Equations
        
        ### Receptor Binding
        
        $$\\frac{dB}{dt} = k_{on}[IC](R_{total} - B) - k_{off}B$$
        
        where $B$ is bound receptors, $[IC]$ is immune complex concentration, 
        $k_{on}$ and $k_{off}$ are association and dissociation rate constants.
        
        ### ITAM Phosphorylation
        
        $$\\frac{d[pITAM]}{dt} = k_{phos}B(1 - [pITAM]) - k_{dephos}[pITAM]$$
        
        ### Syk Activation
        
        $$\\frac{d[Syk^*]}{dt} = k_{Syk}[pITAM](1 - [Syk^*]) - k_{-Syk}[Syk^*]$$
        
        ### PI3K/Akt Pathway
        
        $$\\frac{d[PI3K^*]}{dt} = k_{PI3K}[Syk^*](1 - [PI3K^*]) - k_{-PI3K}[PI3K^*]$$
        
        $$\\frac{d[Akt^*]}{dt} = k_{Akt}[PI3K^*](1 - [Akt^*]) - k_{-Akt}[Akt^*]$$
        
        ### Calcium Dynamics
        
        $$\\frac{d[Ca^{2+}]}{dt} = k_{rel}[IP_3]([Ca^{2+}]_{ER} - [Ca^{2+}]) - k_{uptake}([Ca^{2+}] - [Ca^{2+}]_{basal})$$
        
        ### Phagocytosis Efficiency
        
        $$E_{phag} = \\frac{\\Omega \\cdot S_{net}}{EC_{50} + \\Omega \\cdot S_{net}}$$
        
        where $\\Omega$ is opsonization level and $S_{net}$ is net activating signal.
        
        ### ADCC Specific Lysis
        
        $$Lysis(\\%) = 100 \\times \\frac{[Dead]}{[Target]_{total}}$$
        """)
    
    with tab4:
        st.markdown("""
        ## References
        
        ### Key Publications
        
        1. **Nimmerjahn & Ravetch (2008)**. "Fcγ receptors as regulators of immune responses."
           *Nature Reviews Immunology* 8:34-47.
        
        2. **Guilliams et al. (2014)**. "The function of Fcγ receptors in dendritic cells and macrophages."
           *Nature Reviews Immunology* 14:94-108.
        
        3. **Wang et al. (2015)**. "IgG Fc engineering to modulate antibody effector functions."
           *Protein & Cell* 9:63-73.
        
        4. **Bournazos & Ravetch (2017)**. "Fcγ receptor function and the design of vaccination strategies."
           *Immunity* 47:224-233.
        
        5. **Overdijk et al. (2012)**. "Antibody-mediated phagocytosis contributes to the anti-tumor activity."
           *MAbs* 7:311-321.
        
        ### Reviews
        
        6. **Hogarth & Pietersz (2012)**. "Fc receptor-targeted therapies for the treatment of inflammation, 
           cancer and beyond." *Nature Reviews Drug Discovery* 11:311-331.
        
        7. **Bruhns & Jönsson (2015)**. "Mouse and human FcR effector functions."
           *Immunological Reviews* 268:25-51.
        
        8. **DiLillo & Ravetch (2015)**. "Fc-Receptor Interactions Regulate Both Cytotoxic and 
           Immunomodulatory Therapeutic Antibody Effector Functions." *Cancer Immunology Research* 3:704-713.
        
        ### Computational Models
        
        9. **Pandey et al. (2016)**. "Quantitative modeling of immune signaling networks."
           *Journal of Theoretical Biology* 408:274-289.
        
        10. **Faeder et al. (2009)**. "Rule-based modeling of biochemical systems with BioNetGen."
            *Methods in Molecular Biology* 500:113-167.
        """)


if __name__ == "__main__":
    main()
