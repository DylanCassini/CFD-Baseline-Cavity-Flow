from fipy import CellVariable, FaceVariable, Grid2D, DiffusionTerm, ConvectionTerm, Viewer
from fipy.tools import numerix

L = 1.0
N = 50
dL = L / N
viscosity = 1
U = 1.
pressureRelaxation = 0.8
velocityRelaxation = 0.5
if __name__ == '__main__':
    sweeps = 300
else:
    sweeps = 5

mesh = Grid2D(nx=N, ny=N, dx=dL, dy=dL)

pressure = CellVariable(mesh=mesh, name='pressure')
pressureCorrection = CellVariable(mesh=mesh)
xVelocity = CellVariable(mesh=mesh, name='X velocity')
yVelocity = CellVariable(mesh=mesh, name='Y velocity')

velocity = FaceVariable(mesh=mesh, rank=1)

# Add convection terms for non-inviscid flow
xVelocityEq = ConvectionTerm(coeff=velocity) + DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1., 0.])
yVelocityEq = ConvectionTerm(coeff=velocity) + DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0., 1.])

ap = CellVariable(mesh=mesh, value=1.)
coeff = 1./ ap.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = DiffusionTerm(coeff=coeff) - velocity.divergence

from fipy.variables.faceGradVariable import _FaceGradVariable
volume = CellVariable(mesh=mesh, value=mesh.cellVolumes, name='Volume')
contrvolume=volume.arithmeticFaceValue

xVelocity.constrain(0., mesh.facesRight | mesh.facesLeft | mesh.facesBottom)
xVelocity.constrain(U, mesh.facesTop)
yVelocity.constrain(0., mesh.exteriorFaces)
X, Y = mesh.faceCenters
pressureCorrection.constrain(0., mesh.facesLeft & (Y < dL))

from builtins import range
for sweep in range(sweeps):

    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix()
    xres = xVelocityEq.sweep(var=xVelocity,
                             underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix

    yres = yVelocityEq.sweep(var=yVelocity,
                             underRelaxation=velocityRelaxation)

    ## update the ap coefficient from the matrix diagonal
    ap[:] = -numerix.asarray(xmat.takeDiagonal())

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = _FaceGradVariable(pressure)

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / ap.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])
    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / ap.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])
    velocity[..., mesh.exteriorFaces.value] = 0.
    velocity[0, mesh.facesTop.value] = U

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               ap * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               ap * mesh.cellVolumes)

print('sweep:', sweep, ', x residual:', xres, \
                                 ', y residual', yres, \
                                 ', p residual:', pres, \
                                 ', continuity:', max(abs(rhs)))

viewer = Viewer(vars=(pressure, xVelocity, yVelocity, velocity),
               xmin=0., xmax=1., ymin=0., ymax=1., colorbar='vertical', scale=5)

# Save individual plots
import matplotlib.pyplot as plt
import os

# Create Result Plot directory if it doesn't exist
result_dir = r"FVM\Result Plot"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Save pressure contours
plt.figure(figsize=(8, 6))
pressure_viewer = Viewer(vars=pressure, xmin=0., xmax=1., ymin=0., ymax=1., colorbar='vertical')
pressure_viewer.plot()
plt.title('Pressure Contours')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(result_dir, 'fvm_pressure_contours.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save X velocity
plt.figure(figsize=(8, 6))
x_velocity_viewer = Viewer(vars=xVelocity, xmin=0., xmax=1., ymin=0., ymax=1., colorbar='vertical')
x_velocity_viewer.plot()
plt.title('X Velocity')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(result_dir, 'fvm_x_velocity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save Y velocity
plt.figure(figsize=(8, 6))
y_velocity_viewer = Viewer(vars=yVelocity, xmin=0., xmax=1., ymin=0., ymax=1., colorbar='vertical')
y_velocity_viewer.plot()
plt.title('Y Velocity')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(result_dir, 'fvm_y_velocity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save velocity vectors
plt.figure(figsize=(8, 6))
velocity_viewer = Viewer(vars=velocity, xmin=0., xmax=1., ymin=0., ymax=1., scale=5)
velocity_viewer.plot()
plt.title('Velocity Vectors')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(os.path.join(result_dir, 'fvm_velocity_vectors.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save streamlines
import numpy as np
plt.figure(figsize=(8, 6))

# Get cell centers for streamlines
x_centers = mesh.cellCenters[0].value.reshape(N, N)
y_centers = mesh.cellCenters[1].value.reshape(N, N)

# Get velocity components at cell centers
u_values = xVelocity.value.reshape(N, N)
v_values = yVelocity.value.reshape(N, N)

# Create streamlines
plt.streamplot(x_centers, y_centers, u_values, v_values, 
               density=2, linewidth=1, arrowsize=1.5, color='blue')
plt.title('Streamlines')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(result_dir, 'fvm_streamlines.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Figures saved to 'Result Plot' directory:")
print("- fvm_pressure_contours.png")
print("- fvm_x_velocity.png") 
print("- fvm_y_velocity.png")
print("- fvm_velocity_vectors.png")
print("- fvm_streamlines.png")

