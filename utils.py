#!/usr/bin/env python3
"""
Unified utilities for Lid-Driven Cavity Flow solvers
This module contains both SIMPLE and FEM method implementations
with method-specific function naming to avoid conflicts.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

# =============================================================================
# SIMPLE METHOD IMPLEMENTATION
# =============================================================================

class SIMPLEConfig:
    """Configuration class for SIMPLE algorithm parameters"""
    def __init__(self, nx: int = 41, ny: int = 41, L: float = 1.0, H: float = 1.0,
                 rho: float = 1.0, mu: float = 0.01, max_iter: int = 1000,
                 tolerance: float = 1e-4, dt: float = 0.001,
                 alpha_u: float = 0.1, alpha_v: float = 0.1, alpha_p: float = 0.05,
                 pressure_inner_iter: int = 20):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.H = H
        self.rho = rho
        self.mu = mu
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.dt = dt
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.alpha_p = alpha_p
        self.pressure_inner_iter = pressure_inner_iter
        
        # Derived parameters
        self.dx = L / (nx - 1)
        self.dy = H / (ny - 1)
        self.re = rho * L / mu  # Assuming unit velocity scale

def simple_initialize_staggered_grid(config: SIMPLEConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize staggered grid for velocity and pressure fields
    
    Args:
        config: SIMPLE configuration object
        
    Returns:
        Tuple of (u, v, p) arrays
    """
    u = np.zeros((config.ny, config.nx + 1))  # u-velocity (staggered in x)
    v = np.zeros((config.ny + 1, config.nx))  # v-velocity (staggered in y)
    p = np.zeros((config.ny, config.nx))      # Pressure
    return u, v, p

def simple_apply_lid_driven_cavity_bc(u: np.ndarray, v: np.ndarray, u_lid: float = 1.0) -> None:
    """Apply boundary conditions for lid-driven cavity flow (SIMPLE method)
    
    Args:
        u: u-velocity array
        v: v-velocity array
        u_lid: lid velocity (default: 1.0)
    """
    # u-velocity boundary conditions
    u[0, :] = 0.0      # Bottom wall
    u[-1, :] = u_lid   # Top wall (lid)
    u[:, 0] = 0.0      # Left wall
    u[:, -1] = 0.0     # Right wall
    
    # v-velocity boundary conditions
    v[0, :] = 0.0      # Bottom wall
    v[-1, :] = 0.0     # Top wall
    v[:, 0] = 0.0      # Left wall
    v[:, -1] = 0.0     # Right wall

def simple_apply_custom_bc(u: np.ndarray, v: np.ndarray, bc_dict: Dict) -> None:
    """Apply custom boundary conditions (SIMPLE method)
    
    Args:
        u: u-velocity array
        v: v-velocity array
        bc_dict: Dictionary containing boundary condition specifications
                Format: {'u': {'top': value, 'bottom': value, 'left': value, 'right': value},
                        'v': {'top': value, 'bottom': value, 'left': value, 'right': value}}
    """
    if 'u' in bc_dict:
        u_bc = bc_dict['u']
        if 'bottom' in u_bc: u[0, :] = u_bc['bottom']
        if 'top' in u_bc: u[-1, :] = u_bc['top']
        if 'left' in u_bc: u[:, 0] = u_bc['left']
        if 'right' in u_bc: u[:, -1] = u_bc['right']
    
    if 'v' in bc_dict:
        v_bc = bc_dict['v']
        if 'bottom' in v_bc: v[0, :] = v_bc['bottom']
        if 'top' in v_bc: v[-1, :] = v_bc['top']
        if 'left' in v_bc: v[:, 0] = v_bc['left']
        if 'right' in v_bc: v[:, -1] = v_bc['right']

def simple_solve_u_momentum(u: np.ndarray, v: np.ndarray, p: np.ndarray, config: SIMPLEConfig) -> np.ndarray:
    """Solve u-momentum equation using predictor step (SIMPLE method)
    
    Args:
        u: u-velocity array
        v: v-velocity array
        p: pressure array
        config: SIMPLE configuration object
        
    Returns:
        u_star: predicted u-velocity array
    """
    u_star = u.copy()
    
    for i in range(1, config.ny - 1):
        for j in range(1, config.nx):
            # Convective terms using simple upwind scheme
            u_e = 0.5 * (u[i, j] + u[i, j+1]) if j < config.nx-1 else u[i, j]
            u_w = 0.5 * (u[i, j] + u[i, j-1]) if j > 0 else u[i, j]
            u_n = 0.5 * (u[i, j] + u[i+1, j]) if i < config.ny-1 else u[i, j]
            u_s = 0.5 * (u[i, j] + u[i-1, j]) if i > 0 else u[i, j]
            
            v_ne = 0.5 * (v[i, j] + v[i, j-1]) if j > 0 else 0.0
            v_se = 0.5 * (v[i-1, j] + v[i-1, j-1]) if i > 0 and j > 0 else 0.0
            
            # Convective flux
            conv_x = config.rho * (u_e * u_e - u_w * u_w) / config.dx
            conv_y = config.rho * (v_ne * u_n - v_se * u_s) / config.dy
            u_conv = conv_x + conv_y
            
            # Diffusive terms
            if j > 0 and j < config.nx:
                u_diff_x = config.mu * (u[i, j+1] - 2 * u[i, j] + u[i, j-1]) / config.dx**2
            else:
                u_diff_x = 0.0
                
            u_diff_y = config.mu * (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / config.dy**2
            u_diff = u_diff_x + u_diff_y

            # Pressure gradient
            if j > 0:
                pressure_grad = (p[i, j] - p[i, j-1]) / config.dx
            else:
                pressure_grad = 0.0

            # Update with time stepping
            u_star[i, j] = u[i, j] + config.dt * ((-u_conv + u_diff - pressure_grad) / config.rho)
    
    return u_star

def simple_solve_v_momentum(u: np.ndarray, v: np.ndarray, p: np.ndarray, config: SIMPLEConfig) -> np.ndarray:
    """Solve v-momentum equation using predictor step (SIMPLE method)
    
    Args:
        u: u-velocity array
        v: v-velocity array
        p: pressure array
        config: SIMPLE configuration object
        
    Returns:
        v_star: predicted v-velocity array
    """
    v_star = v.copy()
    
    for i in range(1, config.ny):
        for j in range(1, config.nx - 1):
            # Convective terms using simple upwind scheme
            v_e = 0.5 * (v[i, j] + v[i, j+1]) if j < config.nx-1 else v[i, j]
            v_w = 0.5 * (v[i, j] + v[i, j-1]) if j > 0 else v[i, j]
            v_n = 0.5 * (v[i, j] + v[i+1, j]) if i < config.ny else v[i, j]
            v_s = 0.5 * (v[i, j] + v[i-1, j]) if i > 0 else v[i, j]
            
            u_nw = 0.5 * (u[i, j] + u[i-1, j]) if i > 0 else 0.0
            u_sw = 0.5 * (u[i, j-1] + u[i-1, j-1]) if i > 0 and j > 0 else 0.0
            
            # Convective flux
            conv_x = config.rho * (u_nw * v_e - u_sw * v_w) / config.dx
            conv_y = config.rho * (v_n * v_n - v_s * v_s) / config.dy
            v_conv = conv_x + conv_y
            
            # Diffusive terms
            v_diff_x = config.mu * (v[i, j+1] - 2 * v[i, j] + v[i, j-1]) / config.dx**2
            
            if i > 0 and i < config.ny:
                v_diff_y = config.mu * (v[i+1, j] - 2 * v[i, j] + v[i-1, j]) / config.dy**2
            else:
                v_diff_y = 0.0
                
            v_diff = v_diff_x + v_diff_y

            # Pressure gradient
            if i > 0:
                pressure_grad = (p[i, j] - p[i-1, j]) / config.dy
            else:
                pressure_grad = 0.0

            # Update with time stepping
            v_star[i, j] = v[i, j] + config.dt * ((-v_conv + v_diff - pressure_grad) / config.rho)
    
    return v_star

def simple_solve_pressure_correction(u_star: np.ndarray, v_star: np.ndarray, config: SIMPLEConfig) -> np.ndarray:
    """Solve pressure correction equation (SIMPLE method)
    
    Args:
        u_star: predicted u-velocity array
        v_star: predicted v-velocity array
        config: SIMPLE configuration object
        
    Returns:
        p_prime: pressure correction array
    """
    p_prime = np.zeros((config.ny, config.nx))
    
    for _ in range(config.pressure_inner_iter):
        p_prime_old = p_prime.copy()
        for i in range(1, config.ny - 1):
            for j in range(1, config.nx - 1):
                # Mass imbalance
                mass_imbalance = config.rho * ((u_star[i, j] - u_star[i, j-1]) / config.dx + 
                                             (v_star[i, j] - v_star[i-1, j]) / config.dy)
                
                # Pressure correction equation
                ap = config.rho * (2/config.dx**2 + 2/config.dy**2)
                
                p_prime[i, j] = (1 / ap) * (config.rho * ((p_prime[i, j+1] + p_prime[i, j-1]) / config.dx**2 +
                                                         (p_prime[i+1, j] + p_prime[i-1, j]) / config.dy**2) - mass_imbalance)
    
    return p_prime

def simple_correct_velocity_pressure(u: np.ndarray, v: np.ndarray, p: np.ndarray,
                            u_star: np.ndarray, v_star: np.ndarray, p_prime: np.ndarray,
                            config: SIMPLEConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply velocity and pressure corrections (SIMPLE method)
    
    Args:
        u, v, p: current velocity and pressure arrays
        u_star, v_star: predicted velocity arrays
        p_prime: pressure correction array
        config: SIMPLE configuration object
        
    Returns:
        Tuple of corrected (u, v, p) arrays
    """
    # Update pressure
    p_new = p + config.alpha_p * p_prime
    
    # Update u-velocity
    u_new = u.copy()
    for i in range(1, config.ny - 1):
        for j in range(1, config.nx):
            if j > 0:
                u_new[i, j] = u_star[i, j] - config.dt * (p_prime[i, j] - p_prime[i, j-1]) / (config.rho * config.dx)
    
    # Update v-velocity
    v_new = v.copy()
    for i in range(1, config.ny):
        for j in range(1, config.nx - 1):
            if i > 0:
                v_new[i, j] = v_star[i, j] - config.dt * (p_prime[i, j] - p_prime[i-1, j]) / (config.rho * config.dy)
    
    return u_new, v_new, p_new

def simple_check_convergence(u: np.ndarray, v: np.ndarray, u_old: np.ndarray, v_old: np.ndarray,
                     tolerance: float = 1e-4) -> Tuple[bool, float, float]:
    """Check convergence of velocity fields (SIMPLE method)
    
    Args:
        u, v: current velocity arrays
        u_old, v_old: previous iteration velocity arrays
        tolerance: convergence tolerance
        
    Returns:
        Tuple of (converged, u_residual, v_residual)
    """
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u_res = np.linalg.norm(u - u_old) / (u_norm + 1e-12)
    v_res = np.linalg.norm(v - v_old) / (v_norm + 1e-12)
    
    converged = u_res < tolerance and v_res < tolerance
    return converged, u_res, v_res

def simple_interpolate_to_cell_centers(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate staggered grid velocities to cell centers (SIMPLE method)
    
    Args:
        u: u-velocity on staggered grid
        v: v-velocity on staggered grid
        
    Returns:
        Tuple of (u_center, v_center) interpolated to cell centers
    """
    u_center = 0.5 * (u[:, 1:] + u[:, :-1])
    v_center = 0.5 * (v[1:, :] + v[:-1, :])
    return u_center, v_center

def simple_create_meshgrid(config: SIMPLEConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Create meshgrid for plotting (SIMPLE method)
    
    Args:
        config: SIMPLE configuration object
        
    Returns:
        Tuple of (X, Y) meshgrid arrays
    """
    x = np.linspace(0, config.L, config.nx)
    y = np.linspace(0, config.H, config.ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def simple_plot_velocity_vectors(u_center: np.ndarray, v_center: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         config: SIMPLEConfig, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot velocity vectors (SIMPLE method)
    
    Args:
        u_center, v_center: velocity components at cell centers
        X, Y: meshgrid arrays
        config: SIMPLE configuration object
        save_path: path to save the plot (optional)
        show: whether to display the plot
    """
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, u_center, v_center)
    plt.title('Velocity Vectors (SIMPLE)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, config.L])
    plt.ylim([0, config.H])
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def simple_plot_streamlines(u_center: np.ndarray, v_center: np.ndarray, X: np.ndarray, Y: np.ndarray,
                    config: SIMPLEConfig, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot streamlines (SIMPLE method)
    
    Args:
        u_center, v_center: velocity components at cell centers
        X, Y: meshgrid arrays
        config: SIMPLE configuration object
        save_path: path to save the plot (optional)
        show: whether to display the plot
    """
    plt.figure(figsize=(8, 8))
    plt.streamplot(X, Y, u_center, v_center, density=2)
    plt.title('Streamlines (SIMPLE)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def simple_plot_pressure_contours(p: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          config: SIMPLEConfig, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot pressure contours (SIMPLE method)
    
    Args:
        p: pressure array
        X, Y: meshgrid arrays
        config: SIMPLE configuration object
        save_path: path to save the plot (optional)
        show: whether to display the plot
    """
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, p, levels=50, cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Contours (SIMPLE)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def run_simple_solver(config: SIMPLEConfig, bc_function=None, bc_params=None,
                     verbose: bool = True, plot_interval: int = 50) -> Dict:
    """Run complete SIMPLE algorithm solver
    
    Args:
        config: SIMPLE configuration object
        bc_function: boundary condition function (default: lid-driven cavity)
        bc_params: parameters for boundary condition function
        verbose: whether to print iteration progress
        plot_interval: interval for printing progress
        
    Returns:
        Dictionary containing final solution and convergence info
    """
    # Initialize grids
    u, v, p = simple_initialize_staggered_grid(config)
    
    # Set default boundary condition function
    if bc_function is None:
        bc_function = simple_apply_lid_driven_cavity_bc
        bc_params = bc_params or {}
    
    # Main SIMPLE loop
    for it in range(config.max_iter):
        u_old = u.copy()
        v_old = v.copy()
        
        # Apply boundary conditions
        bc_function(u, v, **bc_params)
        
        # Momentum predictor steps
        u_star = simple_solve_u_momentum(u, v, p, config)
        v_star = simple_solve_v_momentum(u, v, p, config)
        
        # Pressure correction
        p_prime = simple_solve_pressure_correction(u_star, v_star, config)
        
        # Velocity and pressure correction
        u, v, p = simple_correct_velocity_pressure(u, v, p, u_star, v_star, p_prime, config)
        
        # Check convergence
        converged, u_res, v_res = simple_check_convergence(u, v, u_old, v_old, config.tolerance)
        
        if verbose and it % plot_interval == 0:
            print(f"Iteration: {it}, u_residual: {u_res:.6f}, v_residual: {v_res:.6f}")
        
        if converged:
            if verbose:
                print(f"Converged at iteration {it}")
            break
    
    if verbose:
        print(f"Final iteration: {it}, u_residual: {u_res:.6f}, v_residual: {v_res:.6f}")
    
    return {
        'u': u,
        'v': v,
        'p': p,
        'iterations': it,
        'u_residual': u_res,
        'v_residual': v_res,
        'converged': converged
    }

# =============================================================================
# FEM METHOD IMPLEMENTATION
# =============================================================================

class FEMConfig:
    """Configuration class for FEM algorithm parameters"""
    def __init__(self, nx: int = 21, ny: int = 21, L: float = 1.0, H: float = 1.0,
                 rho: float = 1.0, mu: float = 0.01, max_iter: int = 1000,
                 tolerance: float = 1e-4, element_type: str = 'Q1',
                 integration_order: int = 2):
        self.nx = nx
        self.ny = ny
        self.L = L
        self.H = H
        self.rho = rho
        self.mu = mu
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.element_type = element_type
        self.integration_order = integration_order
        
        # Derived parameters
        self.dx = L / (nx - 1)
        self.dy = H / (ny - 1)
        self.re = rho * L / mu  # Reynolds number
        self.n_nodes = nx * ny
        self.n_elements = (nx - 1) * (ny - 1)

class FEMMesh:
    """FEM mesh class for handling nodes and elements"""
    def __init__(self, config: FEMConfig):
        self.config = config
        self.nodes = self._generate_nodes()
        self.elements = self._generate_elements()
        self.boundary_nodes = self._identify_boundary_nodes()
        
    def _generate_nodes(self) -> np.ndarray:
        """Generate node coordinates"""
        x = np.linspace(0, self.config.L, self.config.nx)
        y = np.linspace(0, self.config.H, self.config.ny)
        X, Y = np.meshgrid(x, y)
        nodes = np.column_stack([X.ravel(), Y.ravel()])
        return nodes
    
    def _generate_elements(self) -> np.ndarray:
        """Generate element connectivity (quadrilateral elements)"""
        elements = []
        for j in range(self.config.ny - 1):
            for i in range(self.config.nx - 1):
                # Node indices for quadrilateral element (counter-clockwise)
                n1 = j * self.config.nx + i
                n2 = j * self.config.nx + (i + 1)
                n3 = (j + 1) * self.config.nx + (i + 1)
                n4 = (j + 1) * self.config.nx + i
                elements.append([n1, n2, n3, n4])
        return np.array(elements)
    
    def _identify_boundary_nodes(self) -> Dict[str, np.ndarray]:
        """Identify boundary nodes for different walls"""
        boundary = {}
        
        # Bottom wall (y = 0)
        boundary['bottom'] = np.arange(self.config.nx)
        
        # Top wall (y = H) - lid
        boundary['top'] = np.arange((self.config.ny - 1) * self.config.nx, 
                                   self.config.ny * self.config.nx)
        
        # Left wall (x = 0)
        boundary['left'] = np.arange(0, self.config.n_nodes, self.config.nx)
        
        # Right wall (x = L)
        boundary['right'] = np.arange(self.config.nx - 1, self.config.n_nodes, self.config.nx)
        
        return boundary

def fem_gauss_quadrature_2d(order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Get Gauss quadrature points and weights for 2D integration"""
    if order == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif order == 2:
        xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w = np.array([1.0, 1.0])
    elif order == 3:
        xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("Quadrature order not supported")
    
    # Create 2D quadrature points and weights
    xi_2d = []
    w_2d = []
    for i in range(len(xi)):
        for j in range(len(xi)):
            xi_2d.append([xi[i], xi[j]])
            w_2d.append(w[i] * w[j])
    
    return np.array(xi_2d), np.array(w_2d)

def fem_shape_functions_q1(xi: float, eta: float) -> np.ndarray:
    """Bilinear quadrilateral shape functions (Q1)"""
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),  # N1
        0.25 * (1 + xi) * (1 - eta),  # N2
        0.25 * (1 + xi) * (1 + eta),  # N3
        0.25 * (1 - xi) * (1 + eta)   # N4
    ])
    return N

def fem_shape_function_derivatives_q1(xi: float, eta: float) -> np.ndarray:
    """Derivatives of Q1 shape functions with respect to xi and eta"""
    dN_dxi = np.array([
        -0.25 * (1 - eta),  # dN1/dxi
         0.25 * (1 - eta),  # dN2/dxi
         0.25 * (1 + eta),  # dN3/dxi
        -0.25 * (1 + eta)   # dN4/dxi
    ])
    
    dN_deta = np.array([
        -0.25 * (1 - xi),   # dN1/deta
        -0.25 * (1 + xi),   # dN2/deta
         0.25 * (1 + xi),   # dN3/deta
         0.25 * (1 - xi)    # dN4/deta
    ])
    
    return dN_dxi, dN_deta

def fem_jacobian_matrix(nodes_coords: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
    """Calculate Jacobian matrix and determinant"""
    dN_dxi, dN_deta = fem_shape_function_derivatives_q1(xi, eta)
    
    # Jacobian matrix
    J = np.zeros((2, 2))
    J[0, 0] = np.sum(dN_dxi * nodes_coords[:, 0])   # dx/dxi
    J[0, 1] = np.sum(dN_dxi * nodes_coords[:, 1])   # dy/dxi
    J[1, 0] = np.sum(dN_deta * nodes_coords[:, 0])  # dx/deta
    J[1, 1] = np.sum(dN_deta * nodes_coords[:, 1])  # dy/deta
    
    det_J = np.linalg.det(J)
    
    return J, det_J

def fem_global_derivatives(dN_dxi: np.ndarray, dN_deta: np.ndarray, J_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate global derivatives of shape functions"""
    dN_dx = J_inv[0, 0] * dN_dxi + J_inv[0, 1] * dN_deta
    dN_dy = J_inv[1, 0] * dN_dxi + J_inv[1, 1] * dN_deta
    
    return dN_dx, dN_dy

def fem_assemble_stiffness_matrix(mesh: FEMMesh, config: FEMConfig) -> csr_matrix:
    """Assemble global stiffness matrix for viscous terms"""
    n_nodes = config.n_nodes
    K_global = lil_matrix((n_nodes, n_nodes))
    
    # Gauss quadrature points and weights
    xi_points, weights = fem_gauss_quadrature_2d(config.integration_order)
    
    for elem_idx, element in enumerate(mesh.elements):
        # Element node coordinates
        elem_nodes = mesh.nodes[element]
        
        # Element stiffness matrix
        K_elem = np.zeros((4, 4))
        
        # Numerical integration
        for gp_idx, (xi_eta, weight) in enumerate(zip(xi_points, weights)):
            xi, eta = xi_eta
            
            # Shape function derivatives
            dN_dxi, dN_deta = fem_shape_function_derivatives_q1(xi, eta)
            
            # Jacobian
            J, det_J = fem_jacobian_matrix(elem_nodes, xi, eta)
            J_inv = np.linalg.inv(J)
            
            # Global derivatives
            dN_dx, dN_dy = fem_global_derivatives(dN_dxi, dN_deta, J_inv)
            
            # Stiffness matrix contribution
            for i in range(4):
                for j in range(4):
                    K_elem[i, j] += (dN_dx[i] * dN_dx[j] + dN_dy[i] * dN_dy[j]) * det_J * weight
        
        # Assemble into global matrix
        for i in range(4):
            for j in range(4):
                K_global[element[i], element[j]] += config.mu * K_elem[i, j]
    
    return K_global.tocsr()

def fem_assemble_mass_matrix(mesh: FEMMesh, config: FEMConfig) -> csr_matrix:
    """Assemble global mass matrix"""
    n_nodes = config.n_nodes
    M_global = lil_matrix((n_nodes, n_nodes))
    
    # Gauss quadrature points and weights
    xi_points, weights = fem_gauss_quadrature_2d(config.integration_order)
    
    for elem_idx, element in enumerate(mesh.elements):
        # Element node coordinates
        elem_nodes = mesh.nodes[element]
        
        # Element mass matrix
        M_elem = np.zeros((4, 4))
        
        # Numerical integration
        for gp_idx, (xi_eta, weight) in enumerate(zip(xi_points, weights)):
            xi, eta = xi_eta
            
            # Shape functions
            N = fem_shape_functions_q1(xi, eta)
            
            # Jacobian
            J, det_J = fem_jacobian_matrix(elem_nodes, xi, eta)
            
            # Mass matrix contribution
            for i in range(4):
                for j in range(4):
                    M_elem[i, j] += N[i] * N[j] * det_J * weight
        
        # Assemble into global matrix
        for i in range(4):
            for j in range(4):
                M_global[element[i], element[j]] += config.rho * M_elem[i, j]
    
    return M_global.tocsr()

def fem_assemble_convection_matrix(mesh: FEMMesh, config: FEMConfig, u: np.ndarray, v: np.ndarray) -> csr_matrix:
    """Assemble convection matrix (linearized)"""
    n_nodes = config.n_nodes
    C_global = lil_matrix((n_nodes, n_nodes))
    
    # Gauss quadrature points and weights
    xi_points, weights = fem_gauss_quadrature_2d(config.integration_order)
    
    for elem_idx, element in enumerate(mesh.elements):
        # Element node coordinates
        elem_nodes = mesh.nodes[element]
        
        # Element velocities
        u_elem = u[element]
        v_elem = v[element]
        
        # Element convection matrix
        C_elem = np.zeros((4, 4))
        
        # Numerical integration
        for gp_idx, (xi_eta, weight) in enumerate(zip(xi_points, weights)):
            xi, eta = xi_eta
            
            # Shape functions and derivatives
            N = fem_shape_functions_q1(xi, eta)
            dN_dxi, dN_deta = fem_shape_function_derivatives_q1(xi, eta)
            
            # Jacobian
            J, det_J = fem_jacobian_matrix(elem_nodes, xi, eta)
            J_inv = np.linalg.inv(J)
            
            # Global derivatives
            dN_dx, dN_dy = fem_global_derivatives(dN_dxi, dN_deta, J_inv)
            
            # Interpolated velocities at Gauss point
            u_gp = np.sum(N * u_elem)
            v_gp = np.sum(N * v_elem)
            
            # Convection matrix contribution
            for i in range(4):
                for j in range(4):
                    C_elem[i, j] += N[i] * (u_gp * dN_dx[j] + v_gp * dN_dy[j]) * det_J * weight
        
        # Assemble into global matrix
        for i in range(4):
            for j in range(4):
                C_global[element[i], element[j]] += config.rho * C_elem[i, j]
    
    return C_global.tocsr()

def fem_assemble_gradient_matrix(mesh: FEMMesh, config: FEMConfig, direction: str) -> csr_matrix:
    """Assemble gradient matrix for pressure terms"""
    n_nodes = config.n_nodes
    G_global = lil_matrix((n_nodes, n_nodes))
    
    # Gauss quadrature points and weights
    xi_points, weights = fem_gauss_quadrature_2d(config.integration_order)
    
    for elem_idx, element in enumerate(mesh.elements):
        # Element node coordinates
        elem_nodes = mesh.nodes[element]
        
        # Element gradient matrix
        G_elem = np.zeros((4, 4))
        
        # Numerical integration
        for gp_idx, (xi_eta, weight) in enumerate(zip(xi_points, weights)):
            xi, eta = xi_eta
            
            # Shape functions and derivatives
            N = fem_shape_functions_q1(xi, eta)
            dN_dxi, dN_deta = fem_shape_function_derivatives_q1(xi, eta)
            
            # Jacobian
            J, det_J = fem_jacobian_matrix(elem_nodes, xi, eta)
            J_inv = np.linalg.inv(J)
            
            # Global derivatives
            dN_dx, dN_dy = fem_global_derivatives(dN_dxi, dN_deta, J_inv)
            
            # Gradient matrix contribution
            for i in range(4):
                for j in range(4):
                    if direction == 'x':
                        G_elem[i, j] += N[i] * dN_dx[j] * det_J * weight
                    elif direction == 'y':
                        G_elem[i, j] += N[i] * dN_dy[j] * det_J * weight
        
        # Assemble into global matrix
        for i in range(4):
            for j in range(4):
                G_global[element[i], element[j]] += G_elem[i, j]
    
    return G_global.tocsr()

def fem_apply_lid_driven_cavity_bc(mesh: FEMMesh, u: np.ndarray, v: np.ndarray, u_lid: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply boundary conditions for lid-driven cavity flow"""
    u_bc = u.copy()
    v_bc = v.copy()
    
    # Bottom wall: u = v = 0
    u_bc[mesh.boundary_nodes['bottom']] = 0.0
    v_bc[mesh.boundary_nodes['bottom']] = 0.0
    
    # Top wall (lid): u = u_lid, v = 0
    u_bc[mesh.boundary_nodes['top']] = u_lid
    v_bc[mesh.boundary_nodes['top']] = 0.0
    
    # Left wall: u = v = 0
    u_bc[mesh.boundary_nodes['left']] = 0.0
    v_bc[mesh.boundary_nodes['left']] = 0.0
    
    # Right wall: u = v = 0
    u_bc[mesh.boundary_nodes['right']] = 0.0
    v_bc[mesh.boundary_nodes['right']] = 0.0
    
    return u_bc, v_bc

def fem_apply_boundary_conditions_to_matrix(matrix: csr_matrix, boundary_nodes: np.ndarray, value: float = 0.0) -> csr_matrix:
    """Apply Dirichlet boundary conditions to system matrix"""
    matrix_bc = matrix.tolil()
    
    for node in boundary_nodes:
        # Set row to zero except diagonal
        matrix_bc[node, :] = 0.0
        matrix_bc[node, node] = 1.0
    
    return matrix_bc.tocsr()

def fem_solve_momentum_equations(mesh: FEMMesh, config: FEMConfig, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve momentum equations using FEM"""
    # Assemble matrices
    K = fem_assemble_stiffness_matrix(mesh, config)
    M = fem_assemble_mass_matrix(mesh, config)  
    C_u = fem_assemble_convection_matrix(mesh, config, u, v)
    C_v = fem_assemble_convection_matrix(mesh, config, u, v)
    G_x = fem_assemble_gradient_matrix(mesh, config, 'x')
    G_y = fem_assemble_gradient_matrix(mesh, config, 'y')
    
    # Time step parameter (implicit Euler)
    dt = 0.01
    
    # System matrices for u and v momentum
    A_u = M/dt + K + C_u
    A_v = M/dt + K + C_v
    
    # Right-hand side
    rhs_u = (M/dt).dot(u) - G_x.dot(p)
    rhs_v = (M/dt).dot(v) - G_y.dot(p)
    
    # Apply boundary conditions
    all_boundary_nodes = np.concatenate([
        mesh.boundary_nodes['bottom'],
        mesh.boundary_nodes['top'],
        mesh.boundary_nodes['left'],
        mesh.boundary_nodes['right']
    ])
    
    # Apply BCs to u-momentum
    A_u_bc = fem_apply_boundary_conditions_to_matrix(A_u, all_boundary_nodes)
    rhs_u[mesh.boundary_nodes['bottom']] = 0.0
    rhs_u[mesh.boundary_nodes['top']] = 1.0  # lid velocity
    rhs_u[mesh.boundary_nodes['left']] = 0.0
    rhs_u[mesh.boundary_nodes['right']] = 0.0
    
    # Apply BCs to v-momentum
    A_v_bc = fem_apply_boundary_conditions_to_matrix(A_v, all_boundary_nodes)
    rhs_v[all_boundary_nodes] = 0.0
    
    # Solve systems
    u_new = spsolve(A_u_bc, rhs_u)
    v_new = spsolve(A_v_bc, rhs_v)
    
    return u_new, v_new

def fem_solve_pressure_equation(mesh: FEMMesh, config: FEMConfig, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Solve pressure Poisson equation"""
    # Assemble Laplacian matrix for pressure
    K_p = fem_assemble_stiffness_matrix(mesh, config) / config.mu  # Remove viscosity factor
    
    # Divergence of velocity as RHS
    G_x = fem_assemble_gradient_matrix(mesh, config, 'x')
    G_y = fem_assemble_gradient_matrix(mesh, config, 'y')
    
    rhs_p = -(G_x.T.dot(u) + G_y.T.dot(v))
    
    # Apply pressure boundary condition (set reference pressure at one node)
    K_p_bc = K_p.tolil()
    K_p_bc[0, :] = 0.0
    K_p_bc[0, 0] = 1.0
    rhs_p[0] = 0.0
    
    # Solve pressure equation
    p_new = spsolve(K_p_bc.tocsr(), rhs_p)
    
    return p_new

def fem_check_convergence(u: np.ndarray, v: np.ndarray, u_old: np.ndarray, v_old: np.ndarray, tolerance: float = 1e-4) -> Tuple[bool, float, float]:
    """Check convergence of velocity fields"""
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u_res = np.linalg.norm(u - u_old) / (u_norm + 1e-12)
    v_res = np.linalg.norm(v - v_old) / (v_norm + 1e-12)
    
    converged = u_res < tolerance and v_res < tolerance
    return converged, u_res, v_res

def fem_interpolate_solution(u_nodes: np.ndarray, v_nodes: np.ndarray, p_nodes: np.ndarray, config: FEMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate nodal solution to regular grid for visualization"""
    # Create regular grid
    x = np.linspace(0, config.L, config.nx)
    y = np.linspace(0, config.H, config.ny)
    X, Y = np.meshgrid(x, y)
    
    # Reshape nodal values to grid
    u_grid = u_nodes.reshape((config.ny, config.nx))
    v_grid = v_nodes.reshape((config.ny, config.nx))
    p_grid = p_nodes.reshape((config.ny, config.nx))
    
    return u_grid, v_grid, p_grid

def fem_create_meshgrid(config: FEMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Create meshgrid for plotting"""
    x = np.linspace(0, config.L, config.nx)
    y = np.linspace(0, config.H, config.ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def fem_plot_velocity_vectors(u_grid: np.ndarray, v_grid: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         config: FEMConfig, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot velocity vectors"""
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, u_grid, v_grid)
    plt.title('FEM Velocity Vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, config.L])
    plt.ylim([0, config.H])
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def fem_plot_streamlines(u_grid: np.ndarray, v_grid: np.ndarray, X: np.ndarray, Y: np.ndarray,
                     config: FEMConfig, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot streamlines (FEM method)"""
    plt.figure(figsize=(8, 8))
    plt.streamplot(X, Y, u_grid, v_grid, density=2)
    plt.title('Streamlines (FEM)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def fem_plot_pressure_contours(p_grid: np.ndarray, X: np.ndarray, Y: np.ndarray,
                           config: FEMConfig, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot pressure contours (FEM method)"""
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, p_grid, levels=50, cmap='viridis')
    plt.colorbar(label='Pressure')
    plt.title('Pressure Contours (FEM)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def run_fem_solver(config: FEMConfig, verbose: bool = True, plot_interval: int = 50) -> Dict:
    """Run complete FEM solver
    
    Args:
        config: FEM configuration object
        verbose: whether to print iteration progress
        plot_interval: interval for printing progress
        
    Returns:
        Dictionary containing final solution and convergence info
    """
    # Create mesh
    mesh = FEMMesh(config)
    
    # Initialize solution
    u = np.zeros(config.n_nodes)
    v = np.zeros(config.n_nodes)
    p = np.zeros(config.n_nodes)
    
    # Apply initial boundary conditions
    u, v = fem_apply_lid_driven_cavity_bc(mesh, u, v)
    
    # Relaxation factors for stability
    alpha_u = 0.5  # Under-relaxation for velocity
    alpha_p = 0.1  # Under-relaxation for pressure
    
    # Main iteration loop
    for it in range(config.max_iter):
        u_old = u.copy()
        v_old = v.copy()
        
        # Solve momentum equations with previous pressure
        u_new, v_new = fem_solve_momentum_equations(mesh, config, u, v, p)
        
        # Apply under-relaxation for velocity
        u = alpha_u * u_new + (1 - alpha_u) * u_old
        v = alpha_u * v_new + (1 - alpha_u) * v_old
        
        # Apply boundary conditions after velocity update
        u, v = fem_apply_lid_driven_cavity_bc(mesh, u, v)
        
        # Solve pressure equation with updated velocities
        p_new = fem_solve_pressure_equation(mesh, config, u, v)
        
        # Apply under-relaxation for pressure
        p = alpha_p * p_new + (1 - alpha_p) * p
        
        # Check convergence
        converged, u_res, v_res = fem_check_convergence(u, v, u_old, v_old, config.tolerance)
        
        if verbose and it % plot_interval == 0:
            print(f"Iteration: {it}, u_residual: {u_res:.6f}, v_residual: {v_res:.6f}")
        
        if converged:
            if verbose:
                print(f"Converged at iteration {it}")
            break
    
    if verbose:
        print(f"Final iteration: {it}, u_residual: {u_res:.6f}, v_residual: {v_res:.6f}")
    
    return {
        'u': u,
        'v': v,
        'p': p,
        'iterations': it,
        'u_residual': u_res,
        'v_residual': v_res,
        'converged': converged
    }

# =============================================================================
# COMMON UTILITY FUNCTIONS
# =============================================================================

def create_meshgrid(config, method='simple') -> Tuple[np.ndarray, np.ndarray]:
    """Create meshgrid for plotting (unified function)
    
    Args:
        config: Configuration object (SIMPLEConfig or FEMConfig)
        method: 'simple' or 'fem'
        
    Returns:
        Tuple of (X, Y) meshgrid arrays
    """
    if method.lower() == 'simple':
        return simple_create_meshgrid(config)
    elif method.lower() == 'fem':
        return fem_create_meshgrid(config)
    else:
        raise ValueError("Method must be 'simple' or 'fem'")

def plot_velocity_vectors(u, v, X, Y, config, method='simple', save_path=None, show=True):
    """Plot velocity vectors (unified function)
    
    Args:
        u, v: velocity components
        X, Y: meshgrid arrays
        config: Configuration object
        method: 'simple' or 'fem'
        save_path: path to save the plot (optional)
        show: whether to display the plot
    """
    if method.lower() == 'simple':
        simple_plot_velocity_vectors(u, v, X, Y, config, save_path, show)
    elif method.lower() == 'fem':
        fem_plot_velocity_vectors(u, v, X, Y, config, save_path, show)
    else:
        raise ValueError("Method must be 'simple' or 'fem'")

def plot_streamlines(u, v, X, Y, config, method='simple', save_path=None, show=True):
    """Plot streamlines (unified function)
    
    Args:
        u, v: velocity components
        X, Y: meshgrid arrays
        config: Configuration object
        method: 'simple' or 'fem'
        save_path: path to save the plot (optional)
        show: whether to display the plot
    """
    if method.lower() == 'simple':
        simple_plot_streamlines(u, v, X, Y, config, save_path, show)
    elif method.lower() == 'fem':
        fem_plot_streamlines(u, v, X, Y, config, save_path, show)
    else:
        raise ValueError("Method must be 'simple' or 'fem'")

def plot_pressure_contours(p, X, Y, config, method='simple', save_path=None, show=True):
    """Plot pressure contours (unified function)
    
    Args:
        p: pressure array
        X, Y: meshgrid arrays
        config: Configuration object
        method: 'simple' or 'fem'
        save_path: path to save the plot (optional)
        show: whether to display the plot
    """
    if method.lower() == 'simple':
        simple_plot_pressure_contours(p, X, Y, config, save_path, show)
    elif method.lower() == 'fem':
        fem_plot_pressure_contours(p, X, Y, config, save_path, show)
    else:
        raise ValueError("Method must be 'simple' or 'fem'")

def apply_lid_driven_cavity_bc(u, v, config=None, method='simple', u_lid=1.0):
    """Apply lid-driven cavity boundary conditions (unified function)
    
    Args:
        u, v: velocity arrays
        config: Configuration object (optional for some methods)
        method: 'simple' or 'fem'
        u_lid: lid velocity
        
    Returns:
        For SIMPLE: None (modifies arrays in-place)
        For FEM: Tuple of (u_bc, v_bc)
    """
    if method.lower() == 'simple':
        simple_apply_lid_driven_cavity_bc(u, v, u_lid)
        return None
    elif method.lower() == 'fem':
        # For FEM, we need a mesh object, so this is a simplified version
        # In practice, you should use the method-specific function directly
        raise NotImplementedError("For FEM, use fem_apply_lid_driven_cavity_bc with mesh object")
    else:
        raise ValueError("Method must be 'simple' or 'fem'")

def check_convergence(u, v, u_old, v_old, tolerance=1e-4, method='simple'):
    """Check convergence (unified function)
    
    Args:
        u, v: current velocity arrays
        u_old, v_old: previous iteration velocity arrays
        tolerance: convergence tolerance
        method: 'simple' or 'fem'
        
    Returns:
        Tuple of (converged, u_residual, v_residual)
    """
    if method.lower() == 'simple':
        return simple_check_convergence(u, v, u_old, v_old, tolerance)
    elif method.lower() == 'fem':
        return fem_check_convergence(u, v, u_old, v_old, tolerance)
    else:
        raise ValueError("Method must be 'simple' or 'fem'")

# =============================================================================
# LBM METHOD IMPLEMENTATION
# =============================================================================

class LBMLattice:
    """
    Generic LBM base: memory layout, collide/stream, solid handling.
    Specific stencil (D2Q9) is provided by subclass.
    """

    # Stencil placeholders — subclass overrides
    dct = [(0, 0)]                # lattice directions e_i
    opp = [0]                     # opposite direction index
    w = [1.0]                     # weights

    cs = 1.0 / np.sqrt(3.0)       # lattice sound speed

    # Default physical parameters (lattice units)
    Lx = 1.0                      # domain length in x
    tau = 1.0                     # BGK relaxation time
    t = 0.0                       # simulation time

    def __init__(self, shape, eta=1.0):
        """
        Initialize an (Nx, Ny) lattice with ghost cells on each side.
        """
        self.shape = np.array(shape, dtype=int)           # [Nx, Ny]
        self.ndct = len(self.dct)

        # Precompute index grids for streaming shifts
        def shift_indices(vec):
            return np.meshgrid(
                *[np.roll(np.arange(self.shape[i] + 2), vec[i]) for i in range(2)],
                indexing="ij",
            )

        self.DCT = [shift_indices(e) for e in self.dct]   # indices for each direction

        # Microscopic distributions (with ghost cells)
        self.rho  = np.zeros(self.shape + 2)              # density
        self.vel  = np.zeros((2, *(self.shape + 2)))      # velocity field [2, Nx+2, Ny+2]
        self.f    = np.zeros((self.ndct, *(self.shape + 2)))
        self.fs   = np.zeros((self.ndct, *(self.shape + 2)))
        self.feq  = np.zeros((self.ndct, *(self.shape + 2)))

        # Momentum-exchange accounting (per step, per component)
        self.force_history = [[] for _ in range(2)]

        # Solid mask & adjacency (1=solid, 0=fluid)
        self._solid = np.zeros(self.shape + 2)
        self._SOLID = np.zeros((self.ndct, *(self.shape + 2)))
        self.setup_solid_masks()

        # Viscosity and derived time step
        self._eta = 1.0
        self._dt = 0.0
        self.eta = eta                                     # triggers dt update

        # Boundary hook (override via set_* methods in subclass)
        self.boundary = lambda: None

    # -------- properties (grid metrics & params) --------
    @property
    def Nx(self):
        return int(self.shape[0])

    @property
    def Ny(self):
        return int(self.shape[1])

    @property
    def dx(self):
        return self.Lx / float(self.Nx)

    @property
    def dt(self):
        return self._dt

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, val):
        """
        Set dynamic viscosity and update dt so that:
            dt = (tau - 0.5) * dx^2 * cs^2 / eta
        This matches your original relation.
        """
        self._eta = float(val)
        self._dt = (self.tau - 0.5) * (self.dx ** 2) * (self.cs ** 2) / self._eta

    # -------- solid masks --------
    @property
    def solid(self):
        return 1.0 * self._solid

    @property
    def SOLID(self):
        """
        Interface mask per direction: 1 if current cell is solid and neighbor
        along direction i is fluid (used by bounce-back).
        """
        return 1.0 * self._SOLID

    def setup_solid_masks(self):
        """Build the per-direction solid interface mask from self._solid."""
        self._SOLID = np.array([self._solid - self._solid[self.DCT[i]] for i in range(self.ndct)])
        self._SOLID[self._SOLID < 0] = 0

    def set_solid(self, mask):
        """Replace solid mask; then recompute interface adjacency."""
        self._solid = (mask > 0).astype(float)
        self.setup_solid_masks()

    def add_solid(self, mask):
        """Add (OR) a mask into the existing solid field."""
        combined = self._solid + (mask > 0).astype(float)
        combined[combined > 1] = 1
        self.set_solid(combined)

    # -------- macroscopic updates --------
    def update_density(self):
        self.rho = np.sum(self.f, axis=0)

    def update_velocity(self):
        self.vel[...] = 0.0
        for I in range(self.ndct):
            ex, ey = self.dct[I]
            self.vel[0] += (self.f[I] * ex) / np.where(self.rho == 0, 1.0, self.rho)
            self.vel[1] += (self.f[I] * ey) / np.where(self.rho == 0, 1.0, self.rho)

    def update_equilibrium(self, rho, vel):
        u2 = vel[0] ** 2 + vel[1] ** 2
        for I in range(self.ndct):
            ex, ey = self.dct[I]
            eu = ex * vel[0] + ey * vel[1]
            self.feq[I] = self.w[I] * rho * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * u2)

    # -------- collide/stream --------
    def collide(self):
        """BGK collision on fluid cells, keep f on solids unchanged."""
        fluid = (self._solid == 0)
        for I in range(self.ndct):
            self.fs[I][fluid] = self.f[I][fluid] + (1.0 / self.tau) * (self.feq[I][fluid] - self.f[I][fluid])

    def stream(self):
        """Shift post-collision populations along each lattice direction."""
        for I in range(self.ndct):
            streamed = self.fs[I][self.DCT[I]]
            # Block streaming into solids, but allow on solid interfaces for bounce bookkeeping
            self.f[I] = streamed * (1.0 - self._solid + self._SOLID[I])

    def bounce_back(self):
        """
        Mid-link bounce-back on solid interfaces.
        (Momentum exchange is accumulated into force_history for accounting.)
        """
        for comp in self.force_history:
            comp.append(np.zeros(self.shape + 2))

        for I in range(self.ndct):
            J = self.opp[I]
            mask = (self._SOLID[J] == 1)
            if not np.any(mask):
                continue
            self.fs[I][mask] = self.f[J][mask]
            ex, ey = self.dct[I]
            self.force_history[0][-1] += self.f[J] * mask * ex
            self.force_history[1][-1] += self.f[J] * mask * ey

    # -------- step --------
    def step(self):
        """One LBM time step."""
        self.update_density()
        self.update_velocity()
        self.update_equilibrium(self.rho, self.vel)
        self.bounce_back()
        self.collide()
        self.stream()
        self.boundary()             # subclass-specific BCs
        self.t += self.dt


class LBMD2Q9(LBMLattice):
    """
    Standard D2Q9 stencil with helpers and cavity boundary setup.
    """

    # Discrete velocities e_i (index order kept from original)
    dct = [
        (0, 0),  # 0: rest
        (1, 0),  # 1: east
        (0, 1),  # 2: north
        (-1, 0), # 3: west
        (0, -1), # 4: south
        (1, 1),  # 5: NE
        (-1, 1), # 6: NW
        (-1, -1),# 7: SW
        (1, -1)  # 8: SE
    ]
    opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    w   = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]

    # ---- plotting utilities ----
    @staticmethod
    def _image(dat):
        """Shift and transpose for human-friendly image coordinates."""
        return np.transpose(np.roll(dat, (1, 1), axis=(0, 1)))

    def imshow(self, dat, title="", cmap="hot"):
        plt.imshow(self._image(dat), origin="lower", cmap=cmap)
        plt.colorbar()
        if title:
            plt.title(title)

    def imshow9(self, fields, title=None, include_rest=False):
        indices = range(9) if include_rest else range(1, 9)
        for I in indices:
            ex, ey = self.dct[I]
            plt.subplot(3, 3, (ex + 2) + 3 * (-ey + 2 - 1))
            self.imshow(fields[I], title=f"dir {I}: ({ex},{ey})")
        if title:
            plt.suptitle(title)
        plt.tight_layout()

    def plot_streamlines(self, scale_physical=True):
        """
        Streamlines of velocity; overlays solids.
        If scale_physical, rescales u by dx/dt like original.
        """
        X = self.dx * (-0.5 + np.arange(self.Nx + 2))
        Y = self.dx * np.arange(self.Ny + 2)

        U = self.vel[0]
        V = self.vel[1]
        if scale_physical:
            U = (self.dx / self.dt) * U
            V = (self.dx / self.dt) * V

        plt.streamplot(X, Y, self._image(U), self._image(V), density=1.4, color="black")
        plt.ylim(0, self.dx * self.Ny)
        plt.xlim(0, self.dx * self.Nx)
        plt.imshow(self._image(1 - (self._solid == 1)), extent=[0, self.Lx, 0, self.Lx * self.Ny / self.Nx],
                   alpha=0.5, cmap="gray", origin="lower", aspect="auto")
        plt.gca().set_aspect("equal")

    # ==============================
    # Lid-driven cavity boundary
    # ==============================
    def set_lid_driven_cavity(self, U_lid):
        """
        Configure boundary to lid-driven cavity:
          - Bottom (y=-1), Left (x=-1), Right (x=Nx): stationary no-slip bounce-back
          - Top ghost row (y=Ny): moving wall with u_w = (U_lid, 0) via
            moving-wall bounce-back correction (Ladd).

        Implementation:
          After streaming, for the top boundary we overwrite the incoming
          populations with:
                f_i = f_opp - 2 w_i * rho * (e_i · u_w) / cs^2
          for i in {4,7,8} (directions pointing DOWN into the fluid).
          rho is taken from the adjacent interior fluid row (y = Ny-1).
        """
        self.U_lid = float(U_lid)
        self.boundary = self._lid_boundary

    def _lid_boundary(self):
        Nx, Ny = self.Nx, self.Ny

        # --- Stationary no-slip on bottom and side walls (simple bounce) ---
        # We mirror ghost<->boundary populations like the original Poiseuille,
        # but now for three stationary walls.

        # Bottom wall (y = -1 ghost ↔ y = 0 interior): bounce pairs (2<->4, 5<->7, 6<->8)
        X = np.arange(-1, Nx + 1)
        for I in (2, 5, 6):
            J = self.opp[I]
            self.f[I][X, -1] = self.fs[J][X, -1]          # ghost
            self.f[J][X,  0] = self.fs[I][X,  0]          # interior first row

        # Left wall (x = -1 ghost ↔ x = 0 interior): pairs (1<->3, 5<->6, 8<->7)
        Y = np.arange(-1, Ny + 1)
        for I in (1, 5, 8):
            J = self.opp[I]
            self.f[I][-1, Y] = self.fs[J][-1, Y]
            self.f[J][ 0, Y] = self.fs[I][ 0, Y]

        # Right wall (x = Nx ghost ↔ x = Nx-1 interior): pairs (3<->1, 7<->8, 6<->5)
        for I in (3, 7, 6):
            J = self.opp[I]
            self.f[I][Nx, Y]   = self.fs[J][Nx, Y]
            self.f[J][Nx-1, Y] = self.fs[I][Nx-1, Y]

        # --- Moving lid at top: y = Ny ghost row with u_w = (U_lid, 0) ---
        # Apply moving-wall bounce-back on incoming-to-fluid directions:
        # i in {4:(0,-1), 7:(-1,-1), 8:(1,-1)}
        Ux, Uy = self.U_lid, 0.0
        uw_dot = {
            4: (0 * Ux + -1 * Uy),
            7: (-1 * Ux + -1 * Uy),
            8: (1 * Ux + -1 * Uy),
        }
        # Local density for correction: take from the last interior row (Ny-1)
        rho_top = self.rho[:, Ny-1].copy()

        for I in (4, 7, 8):
            J = self.opp[I]
            # ghost row at y = Ny, interior top row at y = Ny-1
            # Base bounce:
            self.f[I][:, Ny] = self.fs[J][:, Ny]
            # Moving-wall correction:
            corr = -2.0 * self.w[I] * rho_top * (uw_dot[I]) / (self.cs ** 2)
            # Broadcast corr across x:
            self.f[I][:, Ny] += corr

    # -------- convenience initializers --------
    def initialize_equilibrium(self, rho0=1.0, u0=(0.0, 0.0)):
        """
        Set f = feq(rho0, u0) everywhere (including ghosts) for a clean start.
        """
        self.rho[...] = rho0
        self.vel[0][...] = u0[0]
        self.vel[1][...] = u0[1]
        self.update_equilibrium(self.rho, self.vel)
        for I in range(self.ndct):
            self.f[I][...]  = self.feq[I]
            self.fs[I][...] = self.feq[I]


# LBM utility functions with lbm_ prefix
def lbm_run_lid_driven_cavity(
    Nx=128,
    Ny=128,
    U_lid=-1,
    tau=0.8,
    eta=1.0,
    steps=10000,
    plot_every=10000,
):
    """
    Run a lid-driven cavity simulation and plot streamlines periodically.
    Returns the lattice object so you can inspect fields (vx, vy, rho).

    Notes:
      - Keep tau > 0.5 for stability.
      - Smaller U_lid, moderate tau, and sufficient resolution help stability.
      - dt is computed from (tau, dx, eta) per original code convention.
    """
    lat = LBMD2Q9((Nx, Ny))
    lat.Lx = 1.0
    lat.tau = tau
    lat.eta = eta

    lat.initialize_equilibrium(rho0=1.0, u0=(0.0, 0.0))
    lat.set_lid_driven_cavity(U_lid=U_lid)

    for step in range(1, steps + 1):
        lat.step()
        if plot_every and (step % plot_every == 0 or step == steps):
            plt.figure(figsize=(5.4, 5.4))
            lat.plot_streamlines(scale_physical=True)
            plt.title(f"Lid-Driven Cavity — step {step}, t={lat.t:.3e}")
            plt.tight_layout()
            plt.show()

    return lat


def lbm_create_lattice(shape, eta=1.0):
    """Create a D2Q9 LBM lattice
    
    Args:
        shape: Tuple of (Nx, Ny) for lattice dimensions
        eta: Dynamic viscosity
        
    Returns:
        LBMD2Q9 lattice object
    """
    return LBMD2Q9(shape, eta)


def lbm_plot_velocity_field(lattice, scale_physical=True, save_path=None, show=True):
    """Plot velocity field as streamlines
    
    Args:
        lattice: LBMD2Q9 lattice object
        scale_physical: Whether to scale velocity by dx/dt
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(8, 6))
    lattice.plot_streamlines(scale_physical=scale_physical)
    plt.title(f"LBM Velocity Field (t={lattice.t:.3e})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def lbm_plot_density_field(lattice, save_path=None, show=True):
    """Plot density field
    
    Args:
        lattice: LBMD2Q9 lattice object
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(8, 6))
    lattice.imshow(lattice.rho, title=f"LBM Density Field (t={lattice.t:.3e})", cmap="viridis")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def lbm_extract_velocity_field(lattice, scale_physical=True):
    """Extract velocity field from lattice
    
    Args:
        lattice: LBMD2Q9 lattice object
        scale_physical: Whether to scale velocity by dx/dt
        
    Returns:
        Tuple of (U, V) velocity arrays
    """
    U = lattice.vel[0].copy()
    V = lattice.vel[1].copy()
    
    if scale_physical:
        U = (lattice.dx / lattice.dt) * U
        V = (lattice.dx / lattice.dt) * V
        
    return U, V


def lbm_extract_density_field(lattice):
    """Extract density field from lattice
    
    Args:
        lattice: LBMD2Q9 lattice object
        
    Returns:
        Density array
    """
    return lattice.rho.copy()


# Backward compatibility aliases
interpolate_to_cell_centers = simple_interpolate_to_cell_centers
interpolate_solution = fem_interpolate_solution