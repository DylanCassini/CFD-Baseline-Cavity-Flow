#!/usr/bin/env python3
"""
Finite Element Method (FEM) for Lid-Driven Cavity Flow
This script implements a FEM solver for the incompressible Navier-Stokes equations
to solve the lid-driven cavity flow problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

def main():
    print("=== FEM Lid-Driven Cavity Flow ===")
    
    # Create configuration
    config = FEMConfig(
        nx=21, ny=21,
        L=1.0, H=1.0,
        rho=1.0, mu=0.01,
        max_iter=300,
        tolerance=1e-3,
        element_type='Q1',
        integration_order=2
    )
    
    # Solve using the FEM solver
    result = run_fem_solver(config, verbose=True, plot_interval=10)
    
    # Extract results
    u_nodes = result['u']
    v_nodes = result['v']
    p_nodes = result['p']
    
    # Post-processing
    u_grid, v_grid, p_grid = fem_interpolate_solution(result['u'], result['v'], result['p'], config)
    X, Y = fem_create_meshgrid(config)
    
    # Create output directory
    output_dir = "Result Plot"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    fem_plot_velocity_vectors(u_grid, v_grid, X, Y, config, 
                         save_path=os.path.join(output_dir, "fem_velocity_vectors.png"), show=False)
    
    fem_plot_streamlines(u_grid, v_grid, X, Y, config,
                    save_path=os.path.join(output_dir, "fem_streamlines.png"), show=False)
    
    fem_plot_pressure_contours(p_grid, X, Y, config,
                          save_path=os.path.join(output_dir, "fem_pressure_contours.png"), show=False)
    
    print(f"Converged: {result['converged']}")
    print(f"Final residuals - U: {result['u_residual']:.6f}, V: {result['v_residual']:.6f}")
    print("FEM plots saved successfully")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)
    main()