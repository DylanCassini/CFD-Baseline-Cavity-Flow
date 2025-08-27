import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Physics-Informed Neural Network
# -------------------------------
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        # x: [N,2] -> (x,y)
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)   # outputs [u,v,p]


# -------------------------------
# PDE Residuals for Navier–Stokes
# -------------------------------
def navier_stokes_residual(xy, net, nu):
    xy.requires_grad_(True)
    out = net(xy)
    u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]

    grads = torch.ones_like(u)

    # First-order derivatives
    u_x = torch.autograd.grad(u, xy, grads, retain_graph=True, create_graph=True)[0][:,0:1]
    u_y = torch.autograd.grad(u, xy, grads, retain_graph=True, create_graph=True)[0][:,1:2]
    v_x = torch.autograd.grad(v, xy, grads, retain_graph=True, create_graph=True)[0][:,0:1]
    v_y = torch.autograd.grad(v, xy, grads, retain_graph=True, create_graph=True)[0][:,1:2]
    p_x = torch.autograd.grad(p, xy, grads, retain_graph=True, create_graph=True)[0][:,0:1]
    p_y = torch.autograd.grad(p, xy, grads, retain_graph=True, create_graph=True)[0][:,1:2]

    # Second-order derivatives
    u_xx = torch.autograd.grad(u_x, xy, grads, retain_graph=True, create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, xy, grads, retain_graph=True, create_graph=True)[0][:,1:2]
    v_xx = torch.autograd.grad(v_x, xy, grads, retain_graph=True, create_graph=True)[0][:,0:1]
    v_yy = torch.autograd.grad(v_y, xy, grads, retain_graph=True, create_graph=True)[0][:,1:2]

    # Continuity equation
    f_cont = u_x + v_y

    # Momentum equations (steady)
    f_u = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    f_v = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)

    return f_cont, f_u, f_v


# -------------------------------
# Training Setup
# -------------------------------

def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Neural Net: input=(x,y), output=(u,v,p)
    layers = [2, 50, 50, 50, 50, 3]
    net = PINN(layers).to(device)

    # Viscosity (Re=100 -> nu=0.01 if domain=1 and U=1)
    nu = 0.01

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Collocation points (residual points inside domain)
    N_f = 5000
    x_f = np.random.rand(N_f,1)
    y_f = np.random.rand(N_f,1)
    xy_f = torch.tensor(np.hstack((x_f,y_f)), dtype=torch.float32).to(device)

    # Boundary points (walls)
    N_b = 1250
    # top lid (u=1,v=0)
    x_top = np.random.rand(N_b,1)
    y_top = np.ones((N_b,1))
    u_top = np.ones((N_b,1))
    v_top = np.zeros((N_b,1))

    # other walls (u=0,v=0)
    x_wall = np.vstack([
        np.random.rand(N_b,1),
        np.zeros((N_b,1)),
        np.ones((N_b,1))
    ])
    y_wall = np.vstack([
        np.zeros((N_b,1)),
        np.random.rand(N_b,1),
        np.random.rand(N_b,1)
    ])
    u_wall = np.zeros_like(x_wall)
    v_wall = np.zeros_like(y_wall)

    # Combine boundary data
    x_b = np.vstack([x_top, x_wall])
    y_b = np.vstack([y_top, y_wall])
    u_b = np.vstack([u_top, u_wall])
    v_b = np.vstack([v_top, v_wall])

    xy_b = torch.tensor(np.hstack((x_b,y_b)), dtype=torch.float32).to(device)
    u_b = torch.tensor(u_b, dtype=torch.float32).to(device)
    v_b = torch.tensor(v_b, dtype=torch.float32).to(device)

    # Training history tracking
    history = {
        'total_loss': [],
        'pde_loss': [],
        'bc_loss': [],
        'iterations': []
    }

    # Phase 1: Adam optimizer training
    max_iter = 50000
    is_converged = False
    print(f"Starting Adam training for {max_iter} iterations...")
    
    for it in range(max_iter):
        optimizer.zero_grad()

        # PDE residual loss
        f_cont, f_u, f_v = navier_stokes_residual(xy_f, net, nu)
        loss_f = (f_cont**2).mean() + (f_u**2).mean() + (f_v**2).mean()

        # Boundary loss
        out_b = net(xy_b)
        u_pred, v_pred = out_b[:,0:1], out_b[:,1:2]
        loss_b = ((u_pred - u_b)**2).mean() + ((v_pred - v_b)**2).mean()

        # Total loss
        loss = loss_f + loss_b
        loss.backward()
        optimizer.step()

        # Store training history
        if it % 10 == 0:  # Store every 10 iterations to reduce memory
            history['total_loss'].append(loss.item())
            history['pde_loss'].append(loss_f.item())
            history['bc_loss'].append(loss_b.item())
            history['iterations'].append(it)

        if it % 500 == 0:
            print(f"Iter {it:4d}, Loss={loss.item():.4e}, PDE={loss_f.item():.4e}, BC={loss_b.item():.4e}")

        if (loss_f < 1e-3 and loss_b < 1e-3) and (it > 25000):
            print(f"Preliminarily converged at iteration {it}")
            break

        if (loss_f < 1e-7 and loss_b < 1e-7):
            print(f"Converged at iteration {it}")
            is_converged = True
            break
    
    # Phase 2: L-BFGS optimizer training (if not converged)
    if not is_converged:
        print("\nSwitching to L-BFGS optimizer for fine-tuning...")
        optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=10000, lr=1.0, history_size=50, tolerance_grad=1e-9)
        
        # Define closure function for L-BFGS
        def closure():
            optimizer_lbfgs.zero_grad()
            f_cont, f_u, f_v = navier_stokes_residual(xy_f, net, nu)
            loss_f = (f_cont**2).mean() + (f_u**2).mean() + (f_v**2).mean()
            out_b = net(xy_b)
            u_pred, v_pred = out_b[:,0:1], out_b[:,1:2]
            loss_b = ((u_pred - u_b)**2).mean() + ((v_pred - v_b)**2).mean()
            loss = loss_f + loss_b
            loss.backward()
            
            # Store current losses for history tracking
            closure.loss_f = loss_f.item()
            closure.loss_b = loss_b.item()
            closure.total_loss = loss.item()
            
            return loss
        
        # L-BFGS training loop
        lbfgs_iter = 0
        max_lbfgs_iter = 500
        
        for lbfgs_step in range(max_lbfgs_iter):
            # Perform L-BFGS step
            optimizer_lbfgs.step(closure)
            
            # Update iteration counter
            current_iter = max_iter + lbfgs_step
            
            # Store training history every 10 iterations
            if lbfgs_step % 5 == 0:
                history['total_loss'].append(closure.total_loss)
                history['pde_loss'].append(closure.loss_f)
                history['bc_loss'].append(closure.loss_b)
                history['iterations'].append(current_iter)
            
            # Print progress every 10 iterations
            if lbfgs_step % 10 == 0:
                print(f"L-BFGS Iter {lbfgs_step:4d}, Loss={closure.total_loss:.4e}, PDE={closure.loss_f:.4e}, BC={closure.loss_b:.4e}")
            
            # Check convergence
            if (closure.loss_f < 1e-7 and closure.loss_b < 1e-7):
                print(f"Converged at L-BFGS iteration {lbfgs_step}")
                is_converged = True
                break
        
        if is_converged:
            print("Training converged successfully!")
        else:
            print("Training completed without full convergence.")
    
    print("Training completed!")
    return net, history


# -------------------------------
# Visualization Functions
# -------------------------------

def plot_training_history(history, save_dir="Result Plot"):
    """Plot training loss history"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.semilogy(history['iterations'], history['total_loss'], 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.title('Total Loss History')
    plt.grid(True, alpha=0.3)
    
    # PDE and BC losses
    plt.subplot(1, 3, 2)
    plt.semilogy(history['iterations'], history['pde_loss'], 'r-', linewidth=2, label='PDE Loss')
    plt.semilogy(history['iterations'], history['bc_loss'], 'g-', linewidth=2, label='BC Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('PDE vs BC Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss ratio
    plt.subplot(1, 3, 3)
    pde_bc_ratio = np.array(history['pde_loss']) / np.array(history['bc_loss'])
    plt.plot(history['iterations'], pde_bc_ratio, 'm-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('PDE Loss / BC Loss')
    plt.title('Loss Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_dir}/training_history.png")


def plot_solution_fields(net, nx=100, ny=100, save_dir="PINN/Result Plot"):
    """Plot velocity and pressure fields"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(net.parameters()).device
    
    # Create grid for evaluation
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_test = np.column_stack([X.ravel(), Y.ravel()])
    xy_tensor = torch.tensor(xy_test, dtype=torch.float32).to(device)
    
    # Predict solution
    net.eval()
    with torch.no_grad():
        pred = net(xy_tensor)
        u_pred = pred[:, 0].cpu().numpy().reshape(ny, nx)
        v_pred = pred[:, 1].cpu().numpy().reshape(ny, nx)
        p_pred = pred[:, 2].cpu().numpy().reshape(ny, nx)
    
    # Calculate velocity magnitude and vorticity
    velocity_mag = np.sqrt(u_pred**2 + v_pred**2)
    
    # Calculate vorticity (∂v/∂x - ∂u/∂y)
    du_dy = np.gradient(u_pred, axis=0) / (y[1] - y[0])
    dv_dx = np.gradient(v_pred, axis=1) / (x[1] - x[0])
    vorticity = dv_dx - du_dy
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # U velocity
    im1 = axes[0,0].contourf(X, Y, u_pred, levels=20, cmap='RdBu_r')
    axes[0,0].set_title('U Velocity')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    axes[0,0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0,0])
    
    # V velocity
    im2 = axes[0,1].contourf(X, Y, v_pred, levels=20, cmap='RdBu_r')
    axes[0,1].set_title('V Velocity')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    axes[0,1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Pressure
    im3 = axes[0,2].contourf(X, Y, p_pred, levels=20, cmap='viridis')
    axes[0,2].set_title('Pressure')
    axes[0,2].set_xlabel('X')
    axes[0,2].set_ylabel('Y')
    axes[0,2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Velocity magnitude
    im4 = axes[1,0].contourf(X, Y, velocity_mag, levels=20, cmap='plasma')
    axes[1,0].set_title('Velocity Magnitude')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    axes[1,0].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Streamlines
    axes[1,1].streamplot(X, Y, u_pred, v_pred, density=2, color=velocity_mag, cmap='plasma')
    axes[1,1].set_title('Streamlines')
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('Y')
    axes[1,1].set_aspect('equal')
    
    # Vorticity
    im6 = axes[1,2].contourf(X, Y, vorticity, levels=20, cmap='RdBu_r')
    axes[1,2].set_title('Vorticity')
    axes[1,2].set_xlabel('X')
    axes[1,2].set_ylabel('Y')
    axes[1,2].set_aspect('equal')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'solution_fields.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Solution fields plot saved to {save_dir}/solution_fields.png")
    
    # Print solution statistics
    print("\n=== Solution Statistics ===")
    print(f"Max U velocity: {np.max(u_pred):.6f}")
    print(f"Min U velocity: {np.min(u_pred):.6f}")
    print(f"Max V velocity: {np.max(v_pred):.6f}")
    print(f"Min V velocity: {np.min(v_pred):.6f}")
    print(f"Max velocity magnitude: {np.max(velocity_mag):.6f}")
    print(f"Max pressure: {np.max(p_pred):.6f}")
    print(f"Min pressure: {np.min(p_pred):.6f}")
    print(f"Max vorticity: {np.max(vorticity):.6f}")
    print(f"Min vorticity: {np.min(vorticity):.6f}")


if __name__ == "__main__":
    print("=== PINN Lid-Driven Cavity Flow Simulation ===")
    
    # Train the network
    trained_net, training_history = train_pinn()
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(training_history)
    
    # Plot solution fields
    print("\nPlotting solution fields...")
    plot_solution_fields(trained_net)
    
    print("\n=== Simulation completed successfully! ===")
