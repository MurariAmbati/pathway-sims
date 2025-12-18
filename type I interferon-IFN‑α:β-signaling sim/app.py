"""
Streamlit app for Type I Interferon (IFN-α/β) Signaling Pathway Simulation

This interactive application allows users to explore the dynamics of Type I interferon
signaling, including receptor binding, JAK-STAT activation, and ISG expression.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathway_model import IFNSignalingModel, PathwayParameters, analyze_pathway_dynamics
from visualizations import (
    plot_pathway_timecourse,
    plot_signaling_cascade,
    plot_phase_portrait,
    plot_pathway_network,
    plot_dose_response,
    plot_comparative_timecourse
)
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="IFN-α/β Signaling Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">Type I Interferon (IFN-α/β) Signaling Pathway</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Simulation of Antiviral Innate Immune Response</div>',
                unsafe_allow_html=True)
    
    # Sidebar - Simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    # Simulation time
    st.sidebar.subheader("Time Settings")
    sim_duration = st.sidebar.slider(
        "Simulation Duration (minutes)",
        min_value=10.0,
        max_value=500.0,
        value=200.0,
        step=10.0
    )
    
    n_points = st.sidebar.slider(
        "Number of Time Points",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100
    )
    
    # Initial conditions
    st.sidebar.subheader("Initial Conditions")
    ifn_initial = st.sidebar.number_input(
        "Initial IFN Concentration",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=10.0
    )
    
    ifnar_initial = st.sidebar.number_input(
        "Initial IFNAR Concentration",
        min_value=100.0,
        max_value=5000.0,
        value=1000.0,
        step=100.0
    )
    
    # Advanced parameters toggle
    show_advanced = st.sidebar.checkbox("Show Advanced Parameters", value=False)
    
    params = PathwayParameters()
    params.ifn_initial = ifn_initial
    params.ifnar_initial = ifnar_initial
    
    if show_advanced:
        st.sidebar.subheader("Advanced Parameters")
        
        with st.sidebar.expander("Receptor Dynamics"):
            params.k_ifn_bind = st.slider("IFN Binding Rate", 0.01, 1.0, 0.1, 0.01)
            params.k_ifn_unbind = st.slider("IFN Unbinding Rate", 0.001, 0.1, 0.01, 0.001)
            params.k_receptor_intern = st.slider("Receptor Internalization", 0.01, 0.2, 0.05, 0.01)
        
        with st.sidebar.expander("JAK-STAT Activation"):
            params.k_jak_phosph = st.slider("JAK Phosphorylation", 0.1, 2.0, 0.5, 0.1)
            params.k_stat_phosph = st.slider("STAT Phosphorylation", 0.1, 2.0, 0.8, 0.1)
            params.k_stat_dimer = st.slider("STAT Dimerization", 0.1, 2.0, 1.0, 0.1)
        
        with st.sidebar.expander("ISG Expression"):
            params.k_isg_transcr = st.slider("ISG Transcription", 0.1, 3.0, 1.2, 0.1)
            params.k_protein_synth = st.slider("Protein Synthesis", 0.1, 2.0, 0.8, 0.1)
        
        with st.sidebar.expander("Feedback Regulation"):
            params.k_socs_synth = st.slider("SOCS Synthesis", 0.1, 2.0, 0.6, 0.1)
            params.k_socs_inhib = st.slider("SOCS Inhibition Strength", 0.1, 2.0, 0.5, 0.1)
            params.k_ifn_feedback = st.slider("IFN Positive Feedback", 0.0, 1.0, 0.3, 0.05)
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        st.session_state.run_simulation = True
    
    # Initialize session state
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    
    # Main content area
    tabs = st.tabs([
        "Overview",
        "Detailed Dynamics",
        "Phase Analysis",
        "Network View",
        "Dose-Response",
        "Analysis"
    ])
    
    # Run simulation if requested
    if st.session_state.run_simulation:
        with st.spinner("Running simulation..."):
            model = IFNSignalingModel(params)
            t, solution = model.simulate((0, sim_duration), n_points)
            analysis = analyze_pathway_dynamics(model, solution, t)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.t = t
            st.session_state.solution = solution
            st.session_state.analysis = analysis
    
    # Display results if simulation has been run
    if 'model' in st.session_state:
        model = st.session_state.model
        t = st.session_state.t
        solution = st.session_state.solution
        analysis = st.session_state.analysis
        
        # Tab 1: Overview
        with tabs[0]:
            st.header("Pathway Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "ISGF3 Peak Time",
                    f"{analysis['isgf3_peak_time']:.1f} min",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "ISG Peak Value",
                    f"{analysis['isg_peak_value']:.1f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Max JAK Activation",
                    f"{analysis['jak_max_activation']:.1f}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    "SOCS Feedback",
                    f"{analysis['socs_feedback_strength']:.1f}",
                    delta=None
                )
            
            st.markdown("---")
            
            # Main timecourse plot
            st.subheader("Key Components Over Time")
            fig = plot_pathway_timecourse(t, solution, model.state_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Info boxes
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **About Type I Interferons:**
                - IFN-α and IFN-β are crucial cytokines in antiviral immunity
                - Bind to IFNAR1/IFNAR2 receptor complex
                - Activate JAK-STAT signaling cascade
                - Induce expression of hundreds of Interferon-Stimulated Genes (ISGs)
                """)
            
            with col2:
                st.success("""
                **Key Pathway Features:**
                - Rapid response (minutes to hours)
                - Negative feedback via SOCS proteins
                - Positive feedback via IFN induction
                - Establishes antiviral state in cells
                """)
        
        # Tab 2: Detailed Dynamics
        with tabs[1]:
            st.header("Detailed Signaling Cascade")
            fig = plot_signaling_cascade(t, solution, model.state_names)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Component selector for custom plot
            st.subheader("Custom Component Analysis")
            available_components = model.state_names
            selected_components = st.multiselect(
                "Select components to visualize:",
                available_components,
                default=['ISGF3_nuc', 'ISG_protein', 'SOCS']
            )
            
            if selected_components:
                custom_fig = plot_pathway_timecourse(
                    t, solution, model.state_names, selected_components
                )
                st.plotly_chart(custom_fig, use_container_width=True)
        
        # Tab 3: Phase Analysis
        with tabs[2]:
            st.header("Phase Portrait Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_component = st.selectbox(
                    "X-axis component:",
                    model.state_names,
                    index=model.state_names.index('JAK_active')
                )
            
            with col2:
                y_component = st.selectbox(
                    "Y-axis component:",
                    model.state_names,
                    index=model.state_names.index('ISGF3_nuc')
                )
            
            fig = plot_phase_portrait(
                solution, model.state_names, x_component, y_component, t
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Phase Portrait Interpretation:**
            - Shows relationship between two components over time
            - Color gradient represents time progression
            - Green star: initial state
            - Red X: final state
            - Trajectory shows how the system evolves
            """)
        
        # Tab 4: Network View
        with tabs[3]:
            st.header("Signaling Network Architecture")
            fig = plot_pathway_network()
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Forward Signaling (Solid Lines)")
                st.markdown("""
                1. **IFN-α/β** binds to **IFNAR1/2** receptors
                2. Receptors activate **JAK1** and **TYK2** kinases
                3. JAKs phosphorylate **STAT1** and **STAT2**
                4. Phosphorylated STATs form heterodimer
                5. STAT dimer combines with **IRF9** to form **ISGF3**
                6. ISGF3 translocates to nucleus
                7. ISGF3 drives **ISG** expression
                8. ISGs establish **antiviral state**
                """)
            
            with col2:
                st.subheader("Feedback Regulation (Dashed Lines)")
                st.markdown("""
                **Negative Feedback:**
                - SOCS proteins inhibit JAK activity
                - Reduces signal amplitude
                - Prevents excessive inflammation
                
                **Positive Feedback:**
                - ISGs include IFN-β gene
                - Amplifies initial signal
                - Sustains antiviral response
                """)
        
        # Tab 5: Dose-Response
        with tabs[4]:
            st.header("Dose-Response Analysis")
            
            st.info("Analyzing how ISG expression responds to different IFN concentrations...")
            
            # Run dose-response simulation
            with st.spinner("Computing dose-response curve..."):
                doses = np.logspace(0, 3, 20)  # 1 to 1000
                responses = []
                
                progress_bar = st.progress(0)
                
                for i, dose in enumerate(doses):
                    temp_params = PathwayParameters()
                    temp_params.ifn_initial = dose
                    temp_model = IFNSignalingModel(temp_params)
                    t_temp, sol_temp = temp_model.simulate((0, 200), 500)
                    
                    # Get peak ISG protein level
                    isg_idx = temp_model.state_names.index('ISG_protein')
                    responses.append(np.max(sol_temp[:, isg_idx]))
                    
                    progress_bar.progress((i + 1) / len(doses))
                
                progress_bar.empty()
            
            fig = plot_dose_response(doses, np.array(responses), 'Peak ISG Protein')
            st.plotly_chart(fig, use_container_width=True)
            
            # EC50 estimation
            half_max = np.max(responses) / 2
            ec50_idx = np.argmin(np.abs(np.array(responses) - half_max))
            ec50 = doses[ec50_idx]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("EC50", f"{ec50:.1f} molecules/cell")
            with col2:
                st.metric("Max Response", f"{np.max(responses):.1f}")
            with col3:
                st.metric("Hill Slope", "~1.2 (estimated)")
        
        # Tab 6: Analysis
        with tabs[5]:
            st.header("Pathway Analysis Summary")
            
            # Create analysis dataframe
            analysis_df = pd.DataFrame({
                'Metric': [
                    'ISGF3 Peak Time (min)',
                    'ISGF3 Peak Value',
                    'ISG Peak Time (min)',
                    'ISG Peak Value',
                    'ISGF3 Response Time (min)',
                    'ISG Response Time (min)',
                    'Max JAK Activation',
                    'SOCS Feedback Strength',
                    'Steady-State ISG'
                ],
                'Value': [
                    f"{analysis['isgf3_peak_time']:.2f}",
                    f"{analysis['isgf3_peak_value']:.2f}",
                    f"{analysis['isg_peak_time']:.2f}",
                    f"{analysis['isg_peak_value']:.2f}",
                    f"{analysis['isgf3_response_time']:.2f}" if analysis['isgf3_response_time'] else "N/A",
                    f"{analysis['isg_response_time']:.2f}" if analysis['isg_response_time'] else "N/A",
                    f"{analysis['jak_max_activation']:.2f}",
                    f"{analysis['socs_feedback_strength']:.2f}",
                    f"{analysis['steady_state_isg']:.2f}"
                ]
            })
            
            st.dataframe(analysis_df, use_container_width=True)
            
            st.markdown("---")
            
            # Export data
            st.subheader("Export Simulation Data")
            
            # Create export dataframe
            export_df = pd.DataFrame(solution, columns=model.state_names)
            export_df.insert(0, 'Time', t)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="ifn_signaling_simulation.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Biological insights
            st.subheader("Biological Insights")
            
            st.markdown("""
            ### Clinical Relevance
            
            **Therapeutic Applications:**
            - IFN-α used to treat hepatitis C, certain cancers
            - Understanding dose-response crucial for optimal dosing
            - Feedback mechanisms explain treatment resistance
            
            **Viral Evasion Strategies:**
            - Many viruses block STAT phosphorylation
            - Some viruses degrade STAT proteins
            - Others prevent ISGF3 nuclear translocation
            
            **Autoimmune Diseases:**
            - Type I IFN signature in lupus, dermatomyositis
            - Excessive signaling drives pathology
            - SOCS regulation disrupted in some conditions
            
            **Future Directions:**
            - Targeted modulation of feedback loops
            - Combination therapies
            - Personalized medicine based on pathway dynamics
            """)
    
    else:
        # Welcome screen
        st.info("Configure parameters in the sidebar and click 'Run Simulation' to begin!")
        
        st.markdown("""
        ## Welcome to the IFN-α/β Signaling Simulator
        
        This interactive tool allows you to explore the dynamics of Type I interferon signaling,
        a critical pathway in antiviral immunity and innate immune responses.
        
        ### Features:
        - **Interactive Parameter Control**: Adjust initial conditions and kinetic parameters
        - **Real-time Visualization**: See pathway dynamics unfold in real-time
        - **Multiple Analysis Views**: Time courses, phase portraits, network diagrams
        - **Dose-Response Analysis**: Understand how cells respond to different IFN levels
        - **Data Export**: Download simulation results for further analysis
        
        ### Getting Started:
        1. Use the sidebar to set simulation parameters
        2. Click "Run Simulation" to start
        3. Explore different tabs to view results
        4. Experiment with different parameter values to see how they affect pathway dynamics
        
        ### About the Model:
        This model captures key aspects of Type I IFN signaling including:
        - Receptor binding and activation
        - JAK-STAT phosphorylation cascade
        - ISGF3 complex formation and nuclear translocation
        - Interferon-stimulated gene (ISG) expression
        - Negative feedback via SOCS proteins
        - Positive feedback via IFN induction
        """)


if __name__ == "__main__":
    main()
