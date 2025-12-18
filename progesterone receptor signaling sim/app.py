"""
Progesterone Receptor Signaling Simulation
Professional interface for simulating and analyzing PR signaling dynamics
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import (
    ProgesteroneReceptorModel,
    ReceptorParameters,
    TissueSpecificModel,
    DownstreamSignaling,
    CellularResponse,
    CrossTalkPathways,
    GeneTargets
)

from visualizations import (
    ReceptorVisualizer,
    PathwayVisualizer,
    ComparisonVisualizer,
    NetworkVisualizer
)


# Page configuration
st.set_page_config(
    page_title="Progesterone Receptor Signaling",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498DB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">Progesterone Receptor Signaling</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Computational Analysis of PR-Mediated Cellular Responses</div>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Simulation Parameters")
        
        # Tissue type selection
        tissue_type = st.selectbox(
            "Tissue Type",
            ["Generic", "Uterine", "Breast", "Pregnancy"]
        )
        
        # If pregnancy selected, choose trimester
        trimester = None
        if tissue_type == "Pregnancy":
            trimester = st.selectbox("Trimester", [1, 2, 3])
        
        st.divider()
        
        # Progesterone concentration
        p4_concentration = st.slider(
            "Progesterone Concentration (nM)",
            min_value=0.01,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.2f"
        )
        
        # Simulation time
        sim_time = st.slider(
            "Simulation Time (minutes)",
            min_value=60,
            max_value=720,
            value=360,
            step=60
        )
        
        st.divider()
        
        # Advanced parameters
        with st.expander("Advanced Parameters"):
            st.subheader("Receptor Levels")
            pr_a_total = st.number_input("PR-A Total (nM)", value=1000.0, min_value=0.0)
            pr_b_total = st.number_input("PR-B Total (nM)", value=1500.0, min_value=0.0)
            
            st.subheader("Kinetic Parameters")
            k_on = st.number_input("Binding k_on (nM⁻¹ min⁻¹)", value=0.1, min_value=0.0, format="%.3f")
            k_off = st.number_input("Binding k_off (min⁻¹)", value=0.01, min_value=0.0, format="%.3f")
            k_transcription = st.number_input("Transcription Rate (min⁻¹)", value=0.5, min_value=0.0, format="%.3f")
        
        st.divider()
        
        # Run simulation button
        run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)
    
    # Main content
    tabs = st.tabs([
        "Receptor Dynamics",
        "Dose-Response",
        "Gene Expression",
        "Tissue Comparison",
        "Pathway Network",
        "Cellular Responses"
    ])
    
    # Initialize model based on tissue type
    if tissue_type == "Uterine":
        model = TissueSpecificModel.uterine_model()
    elif tissue_type == "Breast":
        model = TissueSpecificModel.breast_model()
    elif tissue_type == "Pregnancy" and trimester:
        model = TissueSpecificModel.pregnancy_model(trimester)
    else:
        # Generic model with custom parameters if provided
        params = ReceptorParameters()
        if 'pr_a_total' in locals():
            params.PR_A_total = pr_a_total
            params.PR_B_total = pr_b_total
            params.k_on = k_on
            params.k_off = k_off
            params.k_transcription = k_transcription
        model = ProgesteroneReceptorModel(params)
    
    # Tab 1: Receptor Dynamics
    with tabs[0]:
        st.header("Receptor Signaling Cascade")
        
        if run_simulation or 'solution' not in st.session_state:
            # Run simulation
            time_points = np.linspace(0, sim_time, 500)
            
            with st.spinner("Running simulation..."):
                solution = model.simulate(p4_concentration, time_points)
                st.session_state.solution = solution
                st.session_state.time_points = time_points
                st.session_state.p4_concentration = p4_concentration
        
        solution = st.session_state.get('solution')
        time_points = st.session_state.get('time_points')
        
        if solution is not None:
            # Display metrics
            metrics = model.calculate_metrics(solution)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Nuclear Receptor", f"{metrics['peak_nuclear_receptor']:.1f} nM")
            with col2:
                st.metric("Time to Half-Max", f"{metrics['time_to_half_max']:.1f} min")
            with col3:
                st.metric("Steady-State Protein", f"{metrics['steady_state_protein']:.1f} nM")
            with col4:
                st.metric("PR-A/PR-B Ratio", f"{metrics['PR_A_to_PR_B_ratio']:.2f}")
            
            st.divider()
            
            # Visualization
            viz = ReceptorVisualizer()
            fig = viz.plot_receptor_states(time_points, solution)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("View Raw Data"):
                df = pd.DataFrame({
                    'Time (min)': time_points,
                    'PR-A Cytoplasmic': solution[:, 0],
                    'PR-B Cytoplasmic': solution[:, 1],
                    'PR-A Nuclear': solution[:, 6],
                    'PR-B Nuclear': solution[:, 7],
                    'PR:DNA': solution[:, 8],
                    'mRNA': solution[:, 10],
                    'Protein': solution[:, 11]
                })
                st.dataframe(df, height=300)
    
    # Tab 2: Dose-Response
    with tabs[1]:
        st.header("Dose-Response Analysis")
        
        # Concentration range
        col1, col2 = st.columns(2)
        with col1:
            min_conc = st.number_input("Min Concentration (nM)", value=0.01, min_value=0.001, format="%.3f")
        with col2:
            max_conc = st.number_input("Max Concentration (nM)", value=100.0, min_value=0.1)
        
        n_points = st.slider("Number of Points", min_value=5, max_value=20, value=10)
        
        if st.button("Calculate Dose-Response", use_container_width=True):
            concentrations = np.logspace(np.log10(min_conc), np.log10(max_conc), n_points)
            time_points = np.linspace(0, 360, 200)
            
            responses = []
            nuclear_responses = []
            protein_responses = []
            
            progress_bar = st.progress(0)
            for i, conc in enumerate(concentrations):
                solution = model.simulate(conc, time_points)
                metrics = model.calculate_metrics(solution)
                responses.append(metrics['steady_state_protein'])
                nuclear_responses.append(metrics['peak_nuclear_receptor'])
                protein_responses.append(metrics['steady_state_protein'])
                progress_bar.progress((i + 1) / n_points)
            
            # Plot dose-response curves
            col1, col2 = st.columns(2)
            
            with col1:
                viz = ReceptorVisualizer()
                fig1 = viz.plot_dose_response(concentrations, nuclear_responses, 
                                             'Peak Nuclear Receptor (nM)')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = viz.plot_dose_response(concentrations, protein_responses,
                                             'Steady-State Protein (nM)')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Calculate EC50
            from scipy.optimize import curve_fit
            
            def hill_equation(x, ec50, hill_coeff, max_resp):
                return max_resp * (x ** hill_coeff) / (ec50 ** hill_coeff + x ** hill_coeff)
            
            try:
                popt, _ = curve_fit(hill_equation, concentrations, protein_responses,
                                   p0=[1.0, 1.0, max(protein_responses)])
                ec50, hill_coeff, max_resp = popt
                
                st.success(f"**EC₅₀ = {ec50:.3f} nM** | Hill Coefficient = {hill_coeff:.2f} | Max Response = {max_resp:.1f} nM")
            except:
                st.info("Could not fit Hill equation to data")
    
    # Tab 3: Gene Expression
    with tabs[2]:
        st.header("Gene Expression Profile")
        
        solution = st.session_state.get('solution')
        
        if solution is not None:
            # Calculate gene expression
            downstream = DownstreamSignaling()
            
            # Use final PR:DNA bound level
            pr_dna_bound = solution[-1, 8]
            
            tissue_map = {
                "Generic": "uterine",
                "Uterine": "uterine",
                "Breast": "breast",
                "Pregnancy": "uterine"
            }
            
            gene_expr = downstream.gene_expression_profile(
                pr_dna_bound,
                tissue_type=tissue_map.get(tissue_type, "uterine")
            )
            
            # Display as bar chart
            gene_df = pd.DataFrame({
                'Gene': list(gene_expr.keys()),
                'Expression': list(gene_expr.values())
            }).sort_values('Expression', ascending=True)
            
            import plotly.express as px
            fig = px.bar(gene_df, x='Expression', y='Gene', orientation='h',
                        title='Target Gene Expression Levels',
                        labels={'Expression': 'Relative Expression Level'},
                        color='Expression',
                        color_continuous_scale='Blues')
            fig.update_layout(height=600, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Gene categories
            st.subheader("Gene Categories")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Pregnancy/Uterine**")
                for gene in GeneTargets.PREGNANCY_GENES:
                    if gene in gene_expr:
                        st.write(f"• {gene}: {gene_expr[gene]:.2f}")
            
            with col2:
                st.markdown("**Breast Development**")
                for gene in GeneTargets.BREAST_GENES:
                    if gene in gene_expr:
                        st.write(f"• {gene}: {gene_expr[gene]:.2f}")
            
            with col3:
                st.markdown("**Proliferation**")
                for gene in GeneTargets.PROLIFERATION_GENES:
                    if gene in gene_expr:
                        st.write(f"• {gene}: {gene_expr[gene]:.2f}")
    
    # Tab 4: Tissue Comparison
    with tabs[3]:
        st.header("Tissue-Specific Signaling Comparison")
        
        if st.button("Run Tissue Comparison", use_container_width=True):
            time_points = np.linspace(0, 360, 200)
            
            # Simulate different tissues
            tissues = {
                "Uterine": TissueSpecificModel.uterine_model(),
                "Breast": TissueSpecificModel.breast_model(),
                "Pregnancy T1": TissueSpecificModel.pregnancy_model(1),
                "Pregnancy T3": TissueSpecificModel.pregnancy_model(3)
            }
            
            tissue_data = {}
            
            with st.spinner("Running tissue comparisons..."):
                for tissue_name, tissue_model in tissues.items():
                    solution = tissue_model.simulate(p4_concentration, time_points)
                    # Extract nuclear receptor concentration
                    tissue_data[tissue_name] = solution[:, 6] + solution[:, 7]
            
            # Plot comparison
            comp_viz = ComparisonVisualizer()
            fig = comp_viz.plot_tissue_comparison(tissue_data, time_points, 
                                                 'Nuclear Receptor Concentration')
            st.plotly_chart(fig, use_container_width=True)
            
            # Pregnancy-specific comparison
            st.subheader("Pregnancy Progression")
            
            trimester_data = {}
            for trimester in [1, 2, 3]:
                pregnancy_model = TissueSpecificModel.pregnancy_model(trimester)
                solution = pregnancy_model.simulate(p4_concentration, time_points)
                trimester_data[trimester] = solution[:, 11]  # Protein expression
            
            fig2 = comp_viz.plot_trimester_comparison(trimester_data, time_points)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 5: Pathway Network
    with tabs[4]:
        st.header("Signaling Network Architecture")
        
        # Network diagram
        net_viz = NetworkVisualizer()
        fig = net_viz.create_network_diagram()
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Pathway activation analysis
        solution = st.session_state.get('solution')
        
        if solution is not None:
            st.subheader("Pathway Activation Analysis")
            
            # Simulate downstream signaling
            downstream = DownstreamSignaling()
            
            # Use membrane-associated PR (approximated from cytoplasmic)
            pr_membrane = (solution[-1, 2] + solution[-1, 3]) * 0.1
            
            rapid_signaling = downstream.rapid_signaling(pr_membrane, sim_time)
            
            # Display pathway activation
            path_viz = PathwayVisualizer()
            fig = path_viz.plot_pathway_activation(rapid_signaling)
            st.plotly_chart(fig, use_container_width=True)
            
            # Crosstalk analysis
            st.subheader("Pathway Crosstalk")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Growth Factor Crosstalk**")
                egf_level = st.slider("EGF Level", 0.0, 100.0, 50.0)
                igf_level = st.slider("IGF Level", 0.0, 100.0, 30.0)
                
                pr_level = (solution[-1, 6] + solution[-1, 7])
                crosstalk = CrossTalkPathways.growth_factor_crosstalk(pr_level, egf_level, igf_level)
                
                st.metric("Phosphorylated PR", f"{crosstalk['phosphorylated_PR']:.2f} nM")
                st.metric("Ligand-Independent Activity", f"{crosstalk['ligand_independent_activity']:.1f}%")
            
            with col2:
                st.markdown("**Anti-Inflammatory Response**")
                pr_dna = solution[-1, 8]
                inflammation = CrossTalkPathways.inflammatory_modulation(pr_dna)
                
                st.metric("NF-κB Suppression", f"{inflammation['NFkB_suppression']:.1f}%")
                st.metric("IL-10 Induction", f"{inflammation['IL10_induction']:.1f}%")
                st.metric("TGF-β Induction", f"{inflammation['TGFbeta_induction']:.1f}%")
    
    # Tab 6: Cellular Responses
    with tabs[5]:
        st.header("Cellular Response Metrics")
        
        solution = st.session_state.get('solution')
        
        if solution is not None:
            # Calculate gene expression for responses
            downstream = DownstreamSignaling()
            pr_dna_bound = solution[-1, 8]
            
            tissue_map = {
                "Generic": "uterine",
                "Uterine": "uterine",
                "Breast": "breast",
                "Pregnancy": "uterine"
            }
            
            gene_expr = downstream.gene_expression_profile(
                pr_dna_bound,
                tissue_type=tissue_map.get(tissue_type, "uterine")
            )
            
            # Calculate cellular responses
            cellular_resp = CellularResponse()
            
            responses = {}
            
            if tissue_type in ["Uterine", "Pregnancy", "Generic"]:
                responses['Decidualization'] = cellular_resp.decidualization_score(gene_expr)
                responses['Implantation Receptivity'] = cellular_resp.implantation_receptivity(gene_expr)
            
            if tissue_type in ["Breast", "Generic"]:
                responses['Mammary Development'] = cellular_resp.mammary_development(gene_expr)
            
            responses['Cell Cycle Progression'] = cellular_resp.cell_cycle_progression(gene_expr)
            responses['Proliferation Index'] = downstream.calculate_proliferation_index(gene_expr)
            
            # Display as radar chart
            path_viz = PathwayVisualizer()
            fig = path_viz.plot_cellular_responses(responses)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Display individual metrics
            st.subheader("Response Metrics")
            
            cols = st.columns(len(responses))
            for i, (response_name, value) in enumerate(responses.items()):
                with cols[i]:
                    # Color-code based on value
                    if value >= 75:
                        delta_color = "normal"
                        status = "High"
                    elif value >= 50:
                        delta_color = "off"
                        status = "Moderate"
                    else:
                        delta_color = "inverse"
                        status = "Low"
                    
                    st.metric(response_name, f"{value:.1f}", delta=status)
            
            # Detailed interpretation
            st.subheader("Biological Interpretation")
            
            if tissue_type == "Uterine" or tissue_type == "Pregnancy":
                st.markdown("""
                **Uterine Function:**
                - Decidualization prepares the endometrium for embryo implantation
                - High implantation receptivity indicates optimal conditions for pregnancy establishment
                - Progesterone maintains pregnancy by regulating immune tolerance and preventing contractions
                """)
            
            if tissue_type == "Breast":
                st.markdown("""
                **Breast Development:**
                - Progesterone drives lobuloalveolar development during pregnancy
                - RANKL and WNT4 signaling promote ductal branching and epithelial proliferation
                - Balanced PR-A/PR-B ratio is critical for normal mammary gland function
                """)
            
            st.markdown("""
            **Cell Proliferation:**
            - Progesterone regulates cell cycle through cyclin and CDK expression
            - Tissue-specific responses depend on PR isoform ratios and cofactor availability
            - Excessive proliferation may contribute to pathological conditions
            """)


if __name__ == "__main__":
    main()
