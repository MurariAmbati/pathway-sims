"""
Advanced pathway analysis tools for IFN-α/β signaling.

Includes sensitivity analysis, parameter estimation, and comparative studies.
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from pathway_model import IFNSignalingModel, PathwayParameters


def parameter_sensitivity_analysis(
    base_params: PathwayParameters,
    param_names: List[str],
    variation_range: float = 0.2,
    n_variations: int = 5,
    sim_duration: float = 200.0
) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying parameters.
    
    Args:
        base_params: Base parameter set
        param_names: List of parameter names to vary
        variation_range: Fractional variation range (e.g., 0.2 = ±20%)
        n_variations: Number of variations per parameter
        sim_duration: Simulation duration
        
    Returns:
        DataFrame with sensitivity results
    """
    results = []
    
    # Base simulation
    base_model = IFNSignalingModel(base_params)
    t_base, sol_base = base_model.simulate((0, sim_duration), 500)
    isg_idx = base_model.state_names.index('ISG_protein')
    base_isg_peak = np.max(sol_base[:, isg_idx])
    
    for param_name in param_names:
        base_value = getattr(base_params, param_name)
        
        for factor in np.linspace(1 - variation_range, 1 + variation_range, n_variations):
            if factor == 1.0:
                continue
                
            # Create modified parameters
            modified_params = PathwayParameters()
            for attr in dir(base_params):
                if not attr.startswith('_'):
                    setattr(modified_params, attr, getattr(base_params, attr))
            
            setattr(modified_params, param_name, base_value * factor)
            
            # Simulate
            model = IFNSignalingModel(modified_params)
            t, sol = model.simulate((0, sim_duration), 500)
            
            # Calculate metrics
            isg_peak = np.max(sol[:, isg_idx])
            sensitivity = (isg_peak - base_isg_peak) / base_isg_peak
            
            results.append({
                'parameter': param_name,
                'variation_factor': factor,
                'variation_percent': (factor - 1) * 100,
                'isg_peak': isg_peak,
                'sensitivity': sensitivity
            })
    
    return pd.DataFrame(results)


def dose_response_curve(
    doses: np.ndarray,
    param_to_vary: str = 'ifn_initial',
    output_metric: str = 'ISG_protein',
    sim_duration: float = 200.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dose-response curve.
    
    Args:
        doses: Array of dose values
        param_to_vary: Parameter name to vary
        output_metric: Output metric to measure
        sim_duration: Simulation duration
        
    Returns:
        Tuple of (doses, responses)
    """
    responses = []
    
    for dose in doses:
        params = PathwayParameters()
        setattr(params, param_to_vary, dose)
        
        model = IFNSignalingModel(params)
        t, sol = model.simulate((0, sim_duration), 500)
        
        idx = model.state_names.index(output_metric)
        response = np.max(sol[:, idx])
        responses.append(response)
    
    return doses, np.array(responses)


def compare_conditions(
    param_sets: List[PathwayParameters],
    labels: List[str],
    sim_duration: float = 200.0
) -> Dict[str, any]:
    """
    Compare multiple parameter conditions.
    
    Args:
        param_sets: List of parameter sets
        labels: Labels for each condition
        sim_duration: Simulation duration
        
    Returns:
        Dictionary containing comparison results
    """
    results = {
        'labels': labels,
        't_list': [],
        'solution_list': [],
        'analysis_list': []
    }
    
    for params in param_sets:
        model = IFNSignalingModel(params)
        t, sol = model.simulate((0, sim_duration), 500)
        
        results['t_list'].append(t)
        results['solution_list'].append(sol)
        
        # Extract key metrics
        isgf3_idx = model.state_names.index('ISGF3_nuc')
        isg_idx = model.state_names.index('ISG_protein')
        
        analysis = {
            'isgf3_peak': np.max(sol[:, isgf3_idx]),
            'isg_peak': np.max(sol[:, isg_idx]),
            'isg_steady_state': sol[-1, isg_idx]
        }
        results['analysis_list'].append(analysis)
    
    return results


def calculate_ec50(doses: np.ndarray, responses: np.ndarray) -> float:
    """
    Calculate EC50 (half-maximal effective concentration).
    
    Args:
        doses: Array of dose values
        responses: Array of response values
        
    Returns:
        EC50 value
    """
    max_response = np.max(responses)
    half_max = max_response / 2
    
    # Find closest response to half-max
    idx = np.argmin(np.abs(responses - half_max))
    ec50 = doses[idx]
    
    return ec50


def temporal_correlation_analysis(
    solution: np.ndarray,
    state_names: List[str],
    components: List[str]
) -> pd.DataFrame:
    """
    Calculate temporal correlation between components.
    
    Args:
        solution: Solution array from simulation
        state_names: List of state variable names
        components: Components to analyze
        
    Returns:
        Correlation matrix as DataFrame
    """
    indices = [state_names.index(comp) for comp in components]
    data = solution[:, indices]
    
    corr_matrix = np.corrcoef(data.T)
    
    return pd.DataFrame(corr_matrix, index=components, columns=components)


def peak_timing_analysis(
    t: np.ndarray,
    solution: np.ndarray,
    state_names: List[str]
) -> pd.DataFrame:
    """
    Analyze peak timing for all components.
    
    Args:
        t: Time array
        solution: Solution array
        state_names: List of state variable names
        
    Returns:
        DataFrame with peak analysis
    """
    results = []
    
    for i, name in enumerate(state_names):
        data = solution[:, i]
        peak_idx = np.argmax(data)
        peak_value = data[peak_idx]
        peak_time = t[peak_idx]
        
        # Calculate time to 10% of peak
        threshold = 0.1 * peak_value
        above_threshold = np.where(data > threshold)[0]
        response_time = t[above_threshold[0]] if len(above_threshold) > 0 else None
        
        results.append({
            'component': name,
            'peak_value': peak_value,
            'peak_time': peak_time,
            'response_time': response_time
        })
    
    return pd.DataFrame(results)


def simulate_feedback_knockouts(
    sim_duration: float = 200.0
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate pathway with feedback mechanisms knocked out.
    
    Args:
        sim_duration: Simulation duration
        
    Returns:
        Dictionary mapping condition to (t, solution) tuples
    """
    results = {}
    
    # Wild-type
    params_wt = PathwayParameters()
    model_wt = IFNSignalingModel(params_wt)
    t_wt, sol_wt = model_wt.simulate((0, sim_duration), 500)
    results['wild_type'] = (t_wt, sol_wt)
    
    # SOCS knockout (no negative feedback)
    params_socs_ko = PathwayParameters()
    params_socs_ko.k_socs_synth = 0.0
    model_socs_ko = IFNSignalingModel(params_socs_ko)
    t_socs_ko, sol_socs_ko = model_socs_ko.simulate((0, sim_duration), 500)
    results['socs_knockout'] = (t_socs_ko, sol_socs_ko)
    
    # IFN feedback knockout (no positive feedback)
    params_ifn_ko = PathwayParameters()
    params_ifn_ko.k_ifn_feedback = 0.0
    model_ifn_ko = IFNSignalingModel(params_ifn_ko)
    t_ifn_ko, sol_ifn_ko = model_ifn_ko.simulate((0, sim_duration), 500)
    results['ifn_feedback_knockout'] = (t_ifn_ko, sol_ifn_ko)
    
    # Double knockout
    params_double_ko = PathwayParameters()
    params_double_ko.k_socs_synth = 0.0
    params_double_ko.k_ifn_feedback = 0.0
    model_double_ko = IFNSignalingModel(params_double_ko)
    t_double_ko, sol_double_ko = model_double_ko.simulate((0, sim_duration), 500)
    results['double_knockout'] = (t_double_ko, sol_double_ko)
    
    return results
