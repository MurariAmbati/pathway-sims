#!/usr/bin/env python3
"""
command-line interface for running simulations
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path

from models import (
    create_neutrophil_receptors,
    create_t_cell_receptors,
    create_monocyte_receptors,
    NEUTROPHIL_PROPS,
    T_CELL_PROPS,
    MONOCYTE_PROPS,
    create_inflammation_tissue,
    create_lymph_node_tissue,
    create_simple_tissue,
    ChemokineGradient,
    CHEMOKINE_LIBRARY,
    LeukocytePopulation
)

from simulation import SimulationEngine, SimulationConfig


def load_config(config_file: str) -> dict:
    """load configuration from yaml file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def setup_simulation_from_config(config: dict) -> SimulationEngine:
    """
    setup simulation from configuration dict
    
    args:
        config: configuration dictionary
    
    returns:
        configured SimulationEngine
    """
    # tissue
    tissue_config = config['tissue']
    size = tuple(tissue_config['size'])
    spacing = tissue_config['grid_spacing']
    
    if tissue_config['type'] == 'inflammation':
        tissue = create_inflammation_tissue(size, spacing)
    elif tissue_config['type'] == 'lymph_node':
        tissue = create_lymph_node_tissue(size, spacing)
    else:
        tissue = create_simple_tissue(size, spacing)
    
    # gradients
    gradients = {}
    for grad_config in config['gradients']:
        chemokine_name = grad_config['name']
        
        if chemokine_name not in CHEMOKINE_LIBRARY:
            print(f"warning: unknown chemokine {chemokine_name}")
            continue
        
        chemokine = CHEMOKINE_LIBRARY[chemokine_name]
        gradient = ChemokineGradient(chemokine, tissue.geometry.grid_size,
                                    tissue.geometry.grid_spacing)
        
        # add source
        if grad_config['source_type'] == 'sphere':
            pos = tuple(grad_config['source_position'])
            gradient.add_sphere_source(
                pos,
                radius=int(grad_config['source_radius']),
                strength=grad_config['source_strength']
            )
        
        # equilibrate
        print(f"equilibrating {chemokine_name} gradient...")
        for _ in range(1000):
            gradient.step(0.1)
        
        # add to dict for each target receptor
        for receptor in grad_config['receptor_targets']:
            gradients[receptor] = gradient
    
    # population
    pop_config = config['population']
    cell_type = pop_config['cell_type']
    
    if cell_type == 'neutrophil':
        props = NEUTROPHIL_PROPS
        receptor_factory = create_neutrophil_receptors
    elif cell_type == 't_cell':
        props = T_CELL_PROPS
        receptor_factory = create_t_cell_receptors
    elif cell_type == 'monocyte':
        props = MONOCYTE_PROPS
        receptor_factory = create_monocyte_receptors
    else:
        raise ValueError(f"unknown cell type: {cell_type}")
    
    population = LeukocytePopulation()
    
    spawn = pop_config['spawn_region']
    spawn_region = (
        (spawn['x'][0], spawn['x'][1]),
        (spawn['y'][0], spawn['y'][1]),
        (spawn['z'][0], spawn['z'][1])
    )
    
    population.spawn_cells(
        n_cells=pop_config['n_cells'],
        properties=props,
        receptors_factory=receptor_factory,
        spawn_region=spawn_region
    )
    
    # simulation config
    sim_config = config['simulation']
    config_obj = SimulationConfig(
        duration=sim_config['duration'],
        dt=sim_config['timestep'],
        save_interval=sim_config['save_interval']
    )
    
    return SimulationEngine(tissue, gradients, population, config_obj)


def save_results(results: dict, output_dir: str):
    """save simulation results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # save metadata
    metadata = {
        'duration': results['duration'],
        'n_cells': results['n_cells'],
        'computation_time': results['computation_time']
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # save time series
    np.save(output_path / 'time.npy', results['time'])
    np.save(output_path / 'mean_speeds.npy', results['mean_speeds'])
    
    # save trajectories
    with open(output_path / 'positions.npy', 'wb') as f:
        np.save(f, results['positions'])
    
    # save states
    with open(output_path / 'states.json', 'w') as f:
        json.dump(results['states'], f)
    
    print(f"\nresults saved to: {output_path}")


def main():
    """main cli entry point"""
    parser = argparse.ArgumentParser(
        description='chemokine receptor signaling simulation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='configuration file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='output directory'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='verbose output'
    )
    
    args = parser.parse_args()
    
    # load config
    print(f"loading configuration from {args.config}")
    config = load_config(args.config)
    
    # setup simulation
    print("setting up simulation...")
    engine = setup_simulation_from_config(config)
    
    # run
    print(f"running simulation (duration: {config['simulation']['duration']}s)...")
    
    def progress(t, p):
        if args.verbose or p in [0.25, 0.5, 0.75, 1.0]:
            print(f"  progress: {p*100:.0f}% (t={t:.1f}s)")
    
    engine.run(progress_callback=progress)
    
    # results
    results = engine.get_results()
    stats = engine.get_summary_statistics()
    
    print("\nsimulation complete!")
    print(f"  cells: {stats['total_cells']}")
    print(f"  mean displacement: {stats['mean_displacement']:.1f} μm")
    print(f"  mean speed: {stats['mean_speed']:.3f} μm/s")
    print(f"  computation time: {results['computation_time']:.2f} s")
    
    # save
    save_results(results, args.output)


if __name__ == "__main__":
    main()
