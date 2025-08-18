"""
Solving lid-driven cavity flow by the SIMPLE algorithm
This script demonstrates how to use the modular functions from utils.py
to solve lid-driven cavity flow and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    SIMPLEConfig, run_simple_solver, simple_interpolate_to_cell_centers,
    simple_create_meshgrid, simple_plot_velocity_vectors, simple_plot_streamlines, simple_plot_pressure_contours,
    simple_apply_lid_driven_cavity_bc, simple_apply_custom_bc
)

def main():
    print("=== Standard Lid-Driven Cavity Flow ===")
    
    # Create configuration
    config = SIMPLEConfig(
        nx=41, ny=41,           # Grid resolution
        L=1.0, H=1.0,           # Domain size
        rho=1.0, mu=0.01,       # Fluid properties
        max_iter=1500,          # Maximum iterations
        tolerance=1e-3,         # Convergence tolerance
        dt=0.001,               # Time step
        alpha_u=0.1,            # U-velocity relaxation
        alpha_v=0.1,            # V-velocity relaxation
        alpha_p=0.05            # Pressure relaxation
    )
    
    # Solve using the modular solver
    result = run_simple_solver(config, verbose=True, plot_interval=100)
    
    # Extract results
    u, v, p = result['u'], result['v'], result['p']
    
    # Post-processing
    u_center, v_center = simple_interpolate_to_cell_centers(u, v)
    X, Y = simple_create_meshgrid(config)
    
    # Create output directory
    output_dir = "Result Plot"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    simple_plot_velocity_vectors(u_center, v_center, X, Y, config, 
                          save_path=os.path.join(output_dir, "simple_velocity_vectors.png"), show=False)
    simple_plot_streamlines(u_center, v_center, X, Y, config, 
                     save_path=os.path.join(output_dir, "simple_streamlines.png"), show=False)
    simple_plot_pressure_contours(p, X, Y, config, 
                           save_path=os.path.join(output_dir, "simple_pressure_contours.png"), show=False)
    
    print(f"Converged: {result['converged']}")
    print(f"Final residuals - U: {result['u_residual']:.6f}, V: {result['v_residual']:.6f}")
    print("Plots saved: velocity_vectors.png, streamlines.png, pressure_contours.png\n")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)
    main()
