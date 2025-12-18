"""
example script demonstrating basic usage
"""

import numpy as np
import matplotlib.pyplot as plt

from models import (
    create_neutrophil_receptors,
    CXCL8,
    ChemokineGradient,
    create_inflammation_tissue,
    NEUTROPHIL_PROPS,
    Leukocyte,
    LeukocytePopulation
)

from simulation import create_neutrophil_recruitment_simulation

from visualization import (
    plot_gradient_2d,
    plot_trajectories_3d,
    plot_population_statistics
)


def example_receptor_binding():
    """
    example 1: receptor-ligand binding kinetics
    """
    print("example 1: receptor binding kinetics")
    print("-" * 50)
    
    # create receptor
    from models.receptors import ChemokineReceptor, CXCR1
    receptor = ChemokineReceptor(CXCR1)
    
    # simulate binding at different concentrations
    concentrations = [1e-9, 5e-9, 10e-9, 50e-9]  # molar
    time = np.arange(0, 100, 0.1)
    
    plt.figure(figsize=(10, 6))
    
    for conc in concentrations:
        receptor = ChemokineReceptor(CXCR1)
        occupancy = []
        
        for t in time:
            receptor.step(conc, 0.1)
            occupancy.append(receptor.occupancy())
        
        plt.plot(time, np.array(occupancy) * 100, 
                label=f'{conc*1e9:.1f} nm', linewidth=2)
    
    plt.xlabel('time (s)')
    plt.ylabel('receptor occupancy (%)')
    plt.title('cxcr1 binding kinetics')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('receptor_binding.png', dpi=150)
    print("saved: receptor_binding.png\n")


def example_gradient_formation():
    """
    example 2: chemokine gradient formation
    """
    print("example 2: gradient formation")
    print("-" * 50)
    
    # create gradient
    grid_size = (50, 50, 50)
    grid_spacing = 2.0
    
    gradient = ChemokineGradient(CXCL8, grid_size, grid_spacing)
    
    # add source
    center = (25, 25, 10)
    gradient.add_sphere_source(center, radius=5, strength=10.0)
    
    # equilibrate
    print("equilibrating gradient...")
    for i in range(1000):
        gradient.step(0.1)
        if i % 200 == 0:
            print(f"  step {i}/1000")
    
    # plot
    fig = plot_gradient_2d(gradient.concentration, slice_axis=2, slice_index=10)
    fig.savefig('gradient_2d.png', dpi=150, bbox_inches='tight')
    print("saved: gradient_2d.png\n")
    
    plt.close()


def example_cell_migration():
    """
    example 3: single cell migration
    """
    print("example 3: single cell migration")
    print("-" * 50)
    
    # setup
    tissue = create_inflammation_tissue(size=(150, 150, 150), grid_spacing=2.0)
    
    gradient = ChemokineGradient(CXCL8, tissue.geometry.grid_size, 
                                tissue.geometry.grid_spacing)
    
    center = (tissue.geometry.grid_size[0] // 2,
             tissue.geometry.grid_size[1] // 2,
             10)
    gradient.add_sphere_source(center, radius=5, strength=15.0)
    
    # equilibrate gradient
    for _ in range(1000):
        gradient.step(0.1)
    
    # create cell
    receptors = create_neutrophil_receptors()
    position = np.array([75.0, 75.0, 140.0])
    cell = Leukocyte(NEUTROPHIL_PROPS, receptors, position)
    
    # simulate
    print("simulating cell migration...")
    gradients = {"CXCR1": gradient, "CXCR2": gradient}
    boundaries = tissue.get_boundaries()
    
    duration = 200.0  # seconds
    dt = 0.1
    n_steps = int(duration / dt)
    
    for i in range(n_steps):
        cell.step(gradients, dt, boundaries)
        if i % 500 == 0:
            print(f"  time: {i*dt:.1f} s")
    
    # plot trajectory
    trajectory = np.array(cell.trajectory)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
           'b-', linewidth=2, alpha=0.7)
    
    # mark start and end
    ax.scatter(*trajectory[0], c='green', s=100, marker='o', label='start')
    ax.scatter(*trajectory[-1], c='red', s=100, marker='*', label='end')
    
    # source location (convert to μm)
    source_pos = np.array(center) * tissue.geometry.grid_spacing
    ax.scatter(*source_pos, c='orange', s=200, marker='x', label='source')
    
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    ax.set_title('neutrophil chemotaxis')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('cell_trajectory.png', dpi=150)
    print("saved: cell_trajectory.png\n")
    
    plt.close()


def example_population_simulation():
    """
    example 4: population simulation
    """
    print("example 4: population simulation")
    print("-" * 50)
    
    # create simulation
    print("creating simulation...")
    engine = create_neutrophil_recruitment_simulation(n_neutrophils=30, 
                                                     duration=200.0)
    
    # run
    print("running simulation...")
    
    def progress(t, p):
        print(f"  progress: {p*100:.0f}% (t={t:.1f}s)")
    
    engine.run(progress_callback=progress)
    
    # get results
    results = engine.get_results()
    stats = engine.get_summary_statistics()
    
    print("\nsummary statistics:")
    print(f"  mean displacement: {stats['mean_displacement']:.1f} μm")
    print(f"  mean speed: {stats['mean_speed']:.3f} μm/s")
    print(f"  computation time: {results['computation_time']:.2f} s")
    
    # plot
    print("\ngenerating plots...")
    
    # trajectories (sample)
    n_plot = min(10, len(engine.population.cells))
    trajectories = [cell.trajectory for cell in engine.population.cells[:n_plot]]
    
    fig_traj = plot_trajectories_3d(trajectories)
    fig_traj.write_html('population_trajectories.html')
    print("saved: population_trajectories.html")
    
    # statistics
    fig_stats = plot_population_statistics(results)
    fig_stats.write_html('population_statistics.html')
    print("saved: population_statistics.html")
    
    print("\nexample complete!")


def run_all_examples():
    """run all examples"""
    print("\n" + "="*70)
    print("chemokine receptor signaling simulation - examples")
    print("="*70 + "\n")
    
    example_receptor_binding()
    example_gradient_formation()
    example_cell_migration()
    example_population_simulation()
    
    print("\n" + "="*70)
    print("all examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
