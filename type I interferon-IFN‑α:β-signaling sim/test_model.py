"""
Test script to verify the IFN signaling model works correctly.
"""

from pathway_model import IFNSignalingModel, PathwayParameters, analyze_pathway_dynamics
import numpy as np


def test_basic_simulation():
    """Test basic simulation runs without errors."""
    print("Testing basic simulation...")
    
    model = IFNSignalingModel()
    t, solution = model.simulate((0, 100), n_points=500)
    
    assert t.shape[0] == 500, "Time array has wrong size"
    assert solution.shape == (500, 18), "Solution array has wrong shape"
    assert np.all(np.isfinite(solution)), "Solution contains NaN or Inf values"
    
    print("✓ Basic simulation passed")


def test_parameter_customization():
    """Test custom parameter simulation."""
    print("Testing custom parameters...")
    
    params = PathwayParameters()
    params.ifn_initial = 200.0
    params.k_jak_phosph = 0.8
    
    model = IFNSignalingModel(params)
    t, solution = model.simulate((0, 100), n_points=500)
    
    assert solution[0, 0] == 200.0, "Initial IFN concentration not set correctly"
    
    print("✓ Parameter customization passed")


def test_pathway_analysis():
    """Test pathway analysis functions."""
    print("Testing pathway analysis...")
    
    model = IFNSignalingModel()
    t, solution = model.simulate((0, 200), n_points=1000)
    
    analysis = analyze_pathway_dynamics(model, solution, t)
    
    required_keys = [
        'isgf3_peak_value', 'isgf3_peak_time',
        'isg_peak_value', 'isg_peak_time',
        'jak_max_activation', 'socs_feedback_strength'
    ]
    
    for key in required_keys:
        assert key in analysis, f"Missing analysis key: {key}"
        assert analysis[key] is not None, f"Analysis key {key} is None"
    
    print("✓ Pathway analysis passed")


def test_component_dynamics():
    """Test that key components behave as expected."""
    print("Testing component dynamics...")
    
    model = IFNSignalingModel()
    t, solution = model.simulate((0, 200), n_points=1000)
    
    # Check that IFN-IFNAR complex forms
    ifn_ifnar_idx = model.state_names.index('IFN_IFNAR')
    assert np.max(solution[:, ifn_ifnar_idx]) > 0, "No IFN-IFNAR complex formed"
    
    # Check that JAKs get activated
    jak_active_idx = model.state_names.index('JAK_active')
    assert np.max(solution[:, jak_active_idx]) > 0, "No JAK activation"
    
    # Check that STATs get phosphorylated
    pstat1_idx = model.state_names.index('pSTAT1')
    assert np.max(solution[:, pstat1_idx]) > 0, "No STAT1 phosphorylation"
    
    # Check that ISGF3 enters nucleus
    isgf3_nuc_idx = model.state_names.index('ISGF3_nuc')
    assert np.max(solution[:, isgf3_nuc_idx]) > 0, "No nuclear ISGF3"
    
    # Check that ISGs are expressed
    isg_protein_idx = model.state_names.index('ISG_protein')
    assert np.max(solution[:, isg_protein_idx]) > 0, "No ISG protein expression"
    
    # Check that SOCS is induced (negative feedback)
    socs_idx = model.state_names.index('SOCS')
    assert np.max(solution[:, socs_idx]) > 0, "No SOCS induction"
    
    print("✓ Component dynamics passed")


def test_conservation_laws():
    """Test approximate conservation laws."""
    print("Testing conservation principles...")
    
    model = IFNSignalingModel()
    t, solution = model.simulate((0, 200), n_points=1000)
    
    # Total STAT1 should be relatively conserved (unphosphorylated + phosphorylated + complexed)
    stat1_idx = model.state_names.index('STAT1')
    pstat1_idx = model.state_names.index('pSTAT1')
    stat12_idx = model.state_names.index('STAT1_STAT2')
    isgf3_cyto_idx = model.state_names.index('ISGF3_cyto')
    isgf3_nuc_idx = model.state_names.index('ISGF3_nuc')
    
    total_stat1 = (solution[:, stat1_idx] + 
                   solution[:, pstat1_idx] + 
                   solution[:, stat12_idx] + 
                   solution[:, isgf3_cyto_idx] + 
                   solution[:, isgf3_nuc_idx])
    
    # Check that total doesn't change too drastically (allow for some synthesis/degradation)
    initial_total = total_stat1[0]
    assert np.all(total_stat1 > 0.5 * initial_total), "STAT1 conservation violated"
    
    print("✓ Conservation principles passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("IFN-α/β Signaling Model Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_basic_simulation()
        test_parameter_customization()
        test_pathway_analysis()
        test_component_dynamics()
        test_conservation_laws()
        
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("The model is working correctly.")
        print("Run 'streamlit run app.py' to start the interactive app.")
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()
