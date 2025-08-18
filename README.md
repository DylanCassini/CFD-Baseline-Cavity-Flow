# CFD Baseline: Lid-Driven Cavity Flow

A comprehensive comparison of traditional and modern Computational Fluid Dynamics (CFD) methods for solving the classic lid-driven cavity flow problem. This project is relatively simple and mainly focused on basic implementation of these methods. That's because it's just for my personal coding practice.

## 🎯 Overview

This project implements and compares **five different CFD approaches** to solve the 2D lid-driven cavity flow problem:

- **FDM**: Finite Difference Method with SIMPLE algorithm
- **FEM**: Finite Element Method with quadrilateral elements
- **FVM**: Finite Volume Method using FiPy
- **LBM**: Lattice Boltzmann Method (D2Q9)
- **PINN**: Physics-Informed Neural Networks

## 📁 Project Structure

```
Lid-Driven Cavity Flow/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── utils.py                     # Shared utilities and functions
├── FDM/                         # Finite Difference Method
│   ├── simple_cavity.py         # SIMPLE algorithm implementation
│   └── Result Plot/             # Generated plots
├── FEM/                         # Finite Element Method
│   ├── fem_cavity.py            # FEM solver
│   └── Result Plot/             # Generated plots
├── FVM/                         # Finite Volume Method
│   ├── fvm_cavity.py            # FiPy-based FVM solver
│   └── Result Plot/             # Generated plots
├── LBM/                         # Lattice Boltzmann Method
│   ├── lbm_cavity.py            # D2Q9 LBM implementation
│   └── Result Plot/             # Generated plots
└── PINN/                        # Physics-Informed Neural Networks
    ├── pinn_cavity.py           # PyTorch PINN implementation
    └── Result Plot/             # Generated plots
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running Individual Solvers

#### 1. Finite Difference Method (SIMPLE)
```bash
cd FDM
python simple_cavity.py
```

#### 2. Finite Element Method
```bash
cd FEM
python fem_cavity.py
```

#### 3. Finite Volume Method
```bash
cd FVM
python fvm_cavity.py
```

#### 4. Lattice Boltzmann Method
```bash
cd LBM
python lbm_cavity.py
```

#### 5. Physics-Informed Neural Networks
```bash
cd PINN
python pinn_cavity.py
```

## 🔬 Problem Description

### Lid-Driven Cavity Flow

The lid-driven cavity is a classic benchmark problem in CFD featuring:

- **Domain**: Unit square [0,1] × [0,1]
- **Boundary Conditions**:
  - Top wall (lid): u = 1, v = 0 (moving lid)
  - Other walls: u = 0, v = 0 (no-slip)
- **Reynolds Number**: Re = 100 (ρUL/μ)
- **Governing Equations**: 2D incompressible Navier-Stokes

### Mathematical Formulation

**Continuity Equation:**
```
∂u/∂x + ∂v/∂y = 0
```

**Momentum Equations:**
```
u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν∇²u
u∂v/∂x + v∂v/∂y = -1/ρ ∂p/∂y + ν∇²v
```

where ν = μ/ρ is the kinematic viscosity.

## 🛠️ Method Descriptions

### 1. Finite Difference Method (FDM)
- **Algorithm**: SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
- **Grid**: Staggered grid for velocity-pressure coupling
- **Features**: Predictor-corrector approach with relaxation
- **Key Files**: `FDM/simple_cavity.py`, `utils.py`

### 2. Finite Element Method (FEM)
- **Elements**: Bilinear quadrilateral (Q1) elements
- **Integration**: Gauss quadrature
- **Features**: Weak form formulation, sparse matrix assembly
- **Key Files**: `FEM/fem_cavity.py`, `utils.py`

### 3. Finite Volume Method (FVM)
- **Library**: FiPy (Python finite volume library)
- **Features**: Built-in convection and diffusion terms
- **Pressure-Velocity Coupling**: SIMPLE-like algorithm
- **Key Files**: `FVM/fvm_cavity.py`

### 4. Lattice Boltzmann Method (LBM)
- **Model**: D2Q9 (2D, 9 velocities)
- **Collision**: BGK approximation
- **Features**: Inherently transient, good for complex geometries
- **Key Files**: `LBM/lbm_cavity.py`

### 5. Physics-Informed Neural Networks (PINN)
- **Framework**: PyTorch
- **Architecture**: Deep neural network with physics constraints
- **Features**: Automatic differentiation for PDE residuals
- **Key Files**: `PINN/pinn_cavity.py`

## 📊 Results and Visualization

Each method generates the following visualizations:

- **Pressure Contours**: Pressure distribution in the cavity
- **Velocity Vectors**: Velocity field visualization
- **Streamlines**: Flow pattern visualization
- **Velocity Profiles**: Centerline velocity profiles (where applicable)

Results are saved in each method's `Result Plot/` directory.

## ⚙️ Configuration

### Default Parameters
- **Grid Resolution**: 41×41 to 50×50 (method dependent)
- **Reynolds Number**: 100
- **Domain**: [0,1] × [0,1]
- **Lid Velocity**: 1.0
- **Convergence Tolerance**: 1e-3 to 1e-4

### Customization
Parameters can be modified in each solver file or through the configuration classes in `utils.py`.

## 🔧 Dependencies

- **Core**: `numpy`, `matplotlib`, `scipy`
- **FVM**: `fipy`
- **PINN**: `torch`
- **LBM**: Custom implementation (no additional deps)
- **Utilities**: `typing` for type hints
