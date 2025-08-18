import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    lbm_run_lid_driven_cavity,
    lbm_create_lattice,
    lbm_plot_velocity_field,
    lbm_plot_density_field,
    lbm_extract_velocity_field,
    lbm_extract_density_field,
    LBMD2Q9
)


def main():
    """
    Main function to run the LBM lid-driven cavity simulation
    """
    print("Starting LBM Lid-Driven Cavity Flow Simulation...")
    
    # Simulation parameters
    Nx = 128          # Grid points in x-direction
    Ny = 128          # Grid points in y-direction
    U_lid = -0.1      # Lid velocity (negative for clockwise circulation)
    tau = 0.85        # BGK relaxation time (must be > 0.5 for stability)
    eta = 1.0         # Dynamic viscosity
    steps = 30000     # Total simulation steps
    plot_every = 30000  # Plot frequency
    
    print(f"Grid size: {Nx} x {Ny}")
    print(f"Lid velocity: {U_lid}")
    print(f"Relaxation time (tau): {tau}")
    print(f"Dynamic viscosity (eta): {eta}")
    print(f"Total steps: {steps}")
    print(f"Plot every: {plot_every} steps")
    print()
    
    # Run the simulation using the unified utils function
    lattice = lbm_run_lid_driven_cavity(
        Nx=Nx,
        Ny=Ny,
        U_lid=U_lid,
        tau=tau,
        eta=eta,
        steps=steps,
        plot_every=plot_every
    )
    
    print(f"Simulation completed at time t = {lattice.t:.6f}")
    print(f"Final time step dt = {lattice.dt:.6e}")
    print(f"Grid spacing dx = {lattice.dx:.6f}")
    print()
    
    # Extract and analyze final results
    U, V = lbm_extract_velocity_field(lattice, scale_physical=True)
    rho = lbm_extract_density_field(lattice)
    
    # Calculate some flow statistics
    velocity_magnitude = np.sqrt(U**2 + V**2)
    max_velocity = np.max(velocity_magnitude)
    avg_density = np.mean(rho)
    density_variation = np.std(rho)
    
    print("Flow Statistics:")
    print(f"Maximum velocity magnitude: {max_velocity:.6f}")
    print(f"Average density: {avg_density:.6f}")
    print(f"Density variation (std): {density_variation:.6e}")
    print()
    
    # Create additional plots
    print("Generating additional plots...")
    
    # Plot final velocity field
    lbm_plot_velocity_field(
        lattice, 
        scale_physical=True, 
        save_path=r"Results Plot\lbm_velocity_field.png",
        show=True
    )
    
    # Plot final density field
    lbm_plot_density_field(
        lattice,
        save_path=r"Results Plot\lbm_density_field.png",
        show=True
    )
    
    # Create a velocity magnitude contour plot
    plt.figure(figsize=(10, 8))
    X = lattice.dx * (-0.5 + np.arange(lattice.Nx + 2))
    Y = lattice.dx * np.arange(lattice.Ny + 2)
    
    # Use the _image method to properly orient the data
    velocity_mag_image = lattice._image(velocity_magnitude)
    X_image, Y_image = np.meshgrid(X, Y)
    
    contour = plt.contourf(X_image, Y_image, velocity_mag_image, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Velocity Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'LBM Velocity Magnitude Contours (t={lattice.t:.3e})')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(r"Results Plot\lbm_velocity_magnitude.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved:")
    print("- lbm_velocity_field.png")
    print("- lbm_density_field.png")
    print("- lbm_velocity_magnitude.png")
    print()
    print("LBM simulation completed successfully!")
    
    return lattice


def run_parameter_study():
    """
    Run a parameter study with different Reynolds numbers
    """
    print("Running LBM parameter study...")

    output_dir = "Result Plot"
    os.makedirs(output_dir, exist_ok=True)
    
    # Different tau values for different Reynolds numbers
    tau_values = [0.6, 0.8, 1.0, 1.2]
    Nx, Ny = 64, 64  # Smaller grid for faster computation
    U_lid = -0.1
    eta = 1.0
    steps = 10000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, tau in enumerate(tau_values):
        print(f"Running simulation with tau = {tau}...")
        
        # Create lattice manually for more control
        lattice = lbm_create_lattice((Nx, Ny), eta=eta)
        lattice.Lx = 1.0
        lattice.tau = tau
        lattice.eta = eta
        
        lattice.initialize_equilibrium(rho0=1.0, u0=(0.0, 0.0))
        lattice.set_lid_driven_cavity(U_lid=U_lid)
        
        # Run simulation without plotting
        for step in range(1, steps + 1):
            lattice.step()
        
        # Plot results
        plt.sca(axes[i])
        lattice.plot_streamlines(scale_physical=True)
        
        # Calculate Reynolds number
        Re = lattice.rho[0, 0] * abs(U_lid) * lattice.Lx / eta  # Approximate Re
        plt.title(f'tau = {tau}, Re ≈ {Re:.1f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lbm_parameter_study.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Parameter study completed!")
    print("Plot saved: lbm_parameter_study.png")


if __name__ == "__main__":
    # Run main simulation
    lattice = main()
    
    # Optionally run parameter study
    print("\n" + "="*50)
    response = input("Run parameter study? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_parameter_study()
    
    print("\nAll simulations completed!")
