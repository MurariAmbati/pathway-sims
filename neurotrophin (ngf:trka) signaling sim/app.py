"""
neurotrophin (ngf/trka) signaling simulation
advanced streamlit interface for pathway modeling and visualization
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.integrate import odeint
import networkx as nx

from pathway_model import NGFTrkAModel
from visualizations import (
    create_network_graph,
    create_temporal_plot,
    create_heatmap,
    create_3d_surface,
    create_phase_portrait
)

# page configuration
st.set_page_config(
    page_title="ngf/trka signaling",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css for impressive ui
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .stApp {
        background: transparent;
    }
    h1, h2, h3 {
        color: #00d4ff;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(143, 0, 255, 0.1));
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # header
    st.markdown("<h1 style='text-align: center;'>neurotrophin signaling simulator</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #8f00ff;'>ngf/trka pathway dynamics</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # sidebar controls
    with st.sidebar:
        st.markdown("## simulation parameters")
        
        st.markdown("### ligand properties")
        ngf_conc = st.slider(
            "ngf concentration (nm)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="nerve growth factor concentration"
        )
        
        st.markdown("### receptor properties")
        trka_density = st.slider(
            "trka receptor density",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="receptors per cell"
        )
        
        st.markdown("### simulation settings")
        time_span = st.slider(
            "simulation time (minutes)",
            min_value=10,
            max_value=240,
            value=60,
            step=10
        )
        
        pathway_focus = st.multiselect(
            "active pathways",
            ["ras/mapk", "pi3k/akt", "plcγ"],
            default=["ras/mapk", "pi3k/akt", "plcγ"]
        )
        
        st.markdown("### advanced options")
        show_feedback = st.checkbox("include feedback loops", value=True)
        show_crosstalk = st.checkbox("show pathway crosstalk", value=True)
        noise_level = st.slider("biological noise (%)", 0, 20, 5)
        
        run_simulation = st.button("run simulation", type="primary")
    
    # main content area
    if run_simulation or 'model' not in st.session_state:
        with st.spinner("simulating molecular dynamics..."):
            # initialize model
            model = NGFTrkAModel(
                ngf_concentration=ngf_conc,
                trka_density=trka_density,
                pathways=pathway_focus,
                feedback=show_feedback,
                crosstalk=show_crosstalk,
                noise=noise_level/100
            )
            
            # run simulation
            time_points = np.linspace(0, time_span, 500)
            results = model.simulate(time_points)
            
            # store results in model for metric calculations
            model.last_results = results
            
            st.session_state.model = model
            st.session_state.results = results
            st.session_state.time_points = time_points
    
    if 'results' in st.session_state:
        results = st.session_state.results
        time_points = st.session_state.time_points
        model = st.session_state.model
        
        # key metrics
        st.markdown("## key pathway metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mapk_max = np.max(results['erk_active']) if 'erk_active' in results else 0
            st.metric(
                "mapk activation",
                f"{mapk_max:.1f}%",
                delta=f"{mapk_max - 50:.1f}%"
            )
        
        with col2:
            akt_max = np.max(results['akt_active']) if 'akt_active' in results else 0
            st.metric(
                "akt activation",
                f"{akt_max:.1f}%",
                delta=f"{akt_max - 50:.1f}%"
            )
        
        with col3:
            survival_score = model.calculate_survival_score()
            st.metric(
                "survival signal",
                f"{survival_score:.1f}%",
                delta="optimal" if survival_score > 70 else "suboptimal"
            )
        
        with col4:
            diff_score = model.calculate_differentiation_score()
            st.metric(
                "differentiation",
                f"{diff_score:.1f}%",
                delta="high" if diff_score > 60 else "moderate"
            )
        
        st.markdown("---")
        
        # visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "pathway network",
            "temporal dynamics",
            "activation heatmap",
            "3d landscape",
            "phase analysis"
        ])
        
        with tab1:
            st.markdown("### signaling network topology")
            fig_network = create_network_graph(model, results)
            st.plotly_chart(fig_network, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### pathway components")
                components_df = model.get_components_dataframe()
                st.dataframe(components_df, use_container_width=True)
            
            with col2:
                st.markdown("#### signal propagation")
                propagation_df = model.get_propagation_metrics()
                st.dataframe(propagation_df, use_container_width=True)
        
        with tab2:
            st.markdown("### temporal activation profiles")
            fig_temporal = create_temporal_plot(time_points, results, pathway_focus)
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            st.markdown("#### dose-response analysis")
            fig_dose = model.create_dose_response_curve()
            st.plotly_chart(fig_dose, use_container_width=True)
        
        with tab3:
            st.markdown("### pathway activation heatmap")
            fig_heatmap = create_heatmap(time_points, results)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("#### correlation matrix")
            fig_corr = model.create_correlation_matrix(results)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            st.markdown("### 3d signaling landscape")
            fig_3d = create_3d_surface(results)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown("#### parameter sensitivity")
            fig_sensitivity = model.perform_sensitivity_analysis()
            st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        with tab5:
            st.markdown("### phase space analysis")
            fig_phase = create_phase_portrait(results)
            st.plotly_chart(fig_phase, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### bifurcation diagram")
                fig_bifurc = model.create_bifurcation_diagram()
                st.plotly_chart(fig_bifurc, use_container_width=True)
            
            with col2:
                st.markdown("#### stability analysis")
                stability_df = model.analyze_stability()
                st.dataframe(stability_df, use_container_width=True)
        
        # export options
        st.markdown("---")
        st.markdown("## export data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = model.export_to_csv(results, time_points)
            st.download_button(
                "download csv",
                csv_data,
                "ngf_trka_simulation.csv",
                "text/csv"
            )
        
        with col2:
            json_data = model.export_to_json(results)
            st.download_button(
                "download json",
                json_data,
                "ngf_trka_simulation.json",
                "application/json"
            )
        
        with col3:
            report_html = model.generate_report(results, time_points)
            st.download_button(
                "download report",
                report_html,
                "ngf_trka_report.html",
                "text/html"
            )

if __name__ == "__main__":
    main()
