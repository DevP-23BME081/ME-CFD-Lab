"""
  CFD & HT Lab - Experiment 4
  Topic: Two-Dimensional Rayleigh-Taylor Instability (VOF FVM)
  
  Student Name : Dev Pateliya
  Roll Number  : 23BME081
  Semester     : 7th Semester (B.Tech Mechanical Engineering)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import factorized

grid_size_x = 48
grid_size_y = 96
domain_length_x = 1.0
domain_length_y = 2.0

grid_spacing_x = domain_length_x / grid_size_x
grid_spacing_y = domain_length_y / grid_size_y

density_light = 1.0
density_heavy = 1.2
gravitational_acceleration = 9.81
kinematic_viscosity = 0.005
reference_density = 0.5 * (density_light + density_heavy)
total_simulation_time = 2.1

grid_center_x = (np.arange(grid_size_x) + 0.5) * grid_spacing_x
grid_center_y = (np.arange(grid_size_y) + 0.5) * grid_spacing_y

initial_interface = 1.0 + 0.04 * np.cos(2 * np.pi * grid_center_x / domain_length_x)
volume_fraction = np.clip((grid_center_y[:, None] + grid_spacing_y / 2 - initial_interface[None, :]) / grid_spacing_y, 0, 1)

velocity_u = np.zeros((grid_size_y, grid_size_x))
velocity_v = np.zeros((grid_size_y + 1, grid_size_x))

operator_x = diags([np.ones(grid_size_x - 1), -2 * np.ones(grid_size_x), np.ones(grid_size_x - 1)], [-1, 0, 1]).tolil()
operator_x[0, -1] = operator_x[-1, 0] = 1

operator_y = diags([np.ones(grid_size_y - 1), -2 * np.ones(grid_size_y), np.ones(grid_size_y - 1)], [-1, 0, 1]).tolil()
operator_y[0, 0] = operator_y[-1, -1] = -1

laplacian_matrix = kron(eye(grid_size_y), operator_x / grid_spacing_x**2) + kron(operator_y / grid_spacing_y**2, eye(grid_size_x))
laplacian_matrix = laplacian_matrix.tolil()
laplacian_matrix[0, :] = 0
laplacian_matrix[0, 0] = 1
solve_pressure = factorized(laplacian_matrix.tocsc())

stored_snapshots = [(0.0, volume_fraction.copy())]
initial_heavy_area = volume_fraction.sum() * grid_spacing_x * grid_spacing_y
current_time = 0.0
total_time_steps = 0
maximum_divergence = 0.0

while current_time < total_simulation_time - 1e-12:
    advection_rate = np.max(np.abs(velocity_u)) / grid_spacing_x + np.max(np.abs(velocity_v)) / grid_spacing_y
    time_step = min(0.003, 0.3 / max(advection_rate, 1e-12),
                     0.45 / (2 * kinematic_viscosity * (1 / grid_spacing_x**2 + 1 / grid_spacing_y**2)), 
                     total_simulation_time - current_time)

    flux_x = velocity_u * np.where(velocity_u >= 0, np.roll(volume_fraction, 1, axis=1), volume_fraction)
    fraction_below = np.vstack((volume_fraction[:1], volume_fraction))
    fraction_above = np.vstack((volume_fraction, volume_fraction[-1:]))
    flux_y = velocity_v * np.where(velocity_v >= 0, fraction_below, fraction_above)
    
    volume_fraction = volume_fraction - time_step * ((np.roll(flux_x, -1, axis=1) - flux_x) / grid_spacing_x + (flux_y[1:] - flux_y[:-1]) / grid_spacing_y)

    u_east_face = 0.5 * (velocity_u + np.roll(velocity_u, -1, axis=1))
    u_west_face = 0.5 * (velocity_u + np.roll(velocity_u, 1, axis=1))
    u_north_face = np.roll(velocity_u, -1, axis=0)
    u_south_face = np.roll(velocity_u, 1, axis=0)
    v_north_face = 0.5 * (velocity_v[1:] + np.roll(velocity_v[1:], 1, axis=1))
    v_south_face = 0.5 * (velocity_v[:-1] + np.roll(velocity_v[:-1], 1, axis=1))
    
    u_north_face[-1] = velocity_u[-1]
    u_south_face[0] = velocity_u[0]
    
    momentum_flux_ue = u_east_face * np.where(u_east_face >= 0, velocity_u, np.roll(velocity_u, -1, axis=1))
    momentum_flux_uw = u_west_face * np.where(u_west_face >= 0, np.roll(velocity_u, 1, axis=1), velocity_u)
    momentum_flux_un = v_north_face * np.where(v_north_face >= 0, velocity_u, u_north_face)
    momentum_flux_us = v_south_face * np.where(v_south_face >= 0, u_south_face, velocity_u)
    
    convection_u = (momentum_flux_ue - momentum_flux_uw) / grid_spacing_x + (momentum_flux_un - momentum_flux_us) / grid_spacing_y
    laplacian_u = (np.roll(velocity_u, -1, axis=1) - 2 * velocity_u + np.roll(velocity_u, 1, axis=1)) / grid_spacing_x**2
    laplacian_u += (u_north_face - 2 * velocity_u + u_south_face) / grid_spacing_y**2
    velocity_u_predicted = velocity_u + time_step * (-convection_u + kinematic_viscosity * laplacian_u)

    velocity_v_interior = velocity_v[1:-1]
    u_east_v = 0.5 * (np.roll(velocity_u[:-1], -1, axis=1) + np.roll(velocity_u[1:], -1, axis=1))
    u_west_v = 0.5 * (velocity_u[:-1] + velocity_u[1:])
    v_north_v = 0.5 * (velocity_v_interior + velocity_v[2:])
    v_south_v = 0.5 * (velocity_v_interior + velocity_v[:-2])
    
    momentum_flux_ve = u_east_v * np.where(u_east_v >= 0, velocity_v_interior, np.roll(velocity_v_interior, -1, axis=1))
    momentum_flux_vw = u_west_v * np.where(u_west_v >= 0, np.roll(velocity_v_interior, 1, axis=1), velocity_v_interior)
    momentum_flux_vn = v_north_v * np.where(v_north_v >= 0, velocity_v_interior, velocity_v[2:])
    momentum_flux_vs = v_south_v * np.where(v_south_v >= 0, velocity_v[:-2], velocity_v_interior)
    
    convection_v = (momentum_flux_ve - momentum_flux_vw) / grid_spacing_x + (momentum_flux_vn - momentum_flux_vs) / grid_spacing_y
    laplacian_v = (np.roll(velocity_v_interior, -1, axis=1) - 2 * velocity_v_interior + np.roll(velocity_v_interior, 1, axis=1)) / grid_spacing_x**2
    laplacian_v += (velocity_v[2:] - 2 * velocity_v_interior + velocity_v[:-2]) / grid_spacing_y**2
    
    buoyancy_force = -gravitational_acceleration * (density_heavy - density_light) * (volume_fraction[:-1] + volume_fraction[1:] - 1) / (2 * reference_density)
    velocity_v_predicted = velocity_v.copy()
    velocity_v_predicted[1:-1] = velocity_v_interior + time_step * (-convection_v + kinematic_viscosity * laplacian_v + buoyancy_force)

    velocity_divergence = (np.roll(velocity_u_predicted, -1, axis=1) - velocity_u_predicted) / grid_spacing_x
    velocity_divergence += (velocity_v_predicted[1:] - velocity_v_predicted[:-1]) / grid_spacing_y
    poisson_rhs = (reference_density / time_step * velocity_divergence).ravel()
    poisson_rhs[0] = 0
    
    pressure_field = solve_pressure(poisson_rhs).reshape(grid_size_y, grid_size_x)
    velocity_u = velocity_u_predicted - (time_step / (reference_density * grid_spacing_x)) * (pressure_field - np.roll(pressure_field, 1, axis=1))
    velocity_v = velocity_v_predicted
    velocity_v[1:-1] -= (time_step / (reference_density * grid_spacing_y)) * (pressure_field[1:] - pressure_field[:-1])

    current_divergence = (np.roll(velocity_u, -1, axis=1) - velocity_u) / grid_spacing_x + (velocity_v[1:] - velocity_v[:-1]) / grid_spacing_y
    maximum_divergence = max(maximum_divergence, np.max(np.abs(current_divergence)))
    
    if volume_fraction.min() < -1e-9 or volume_fraction.max() > 1 + 1e-9:
        raise RuntimeError("Volume fraction went out of physical bounds")
        
    current_time += time_step
    total_time_steps += 1
    
    if current_time >= len(stored_snapshots) * (total_simulation_time / 4) - 1e-12:
        stored_snapshots.append((current_time, volume_fraction.copy()))

final_heavy_area = volume_fraction.sum() * grid_spacing_x * grid_spacing_y

fig, axes = plt.subplots(1, len(stored_snapshots), figsize=(13, 4.8), sharex=True, sharey=True)
for axis, (time_instant, snapshot_field) in zip(axes, stored_snapshots):
    axis.imshow(snapshot_field, origin="lower", extent=(0, domain_length_x, 0, domain_length_y), cmap="Blues", vmin=0, vmax=1)
    axis.contour(grid_center_x, grid_center_y, snapshot_field, levels=[0.5], colors="k", linewidths=0.8)
    axis.set_title(f"t = {time_instant:.2f} s")
    axis.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
fig.suptitle("Experiment 4: Rayleigh-Taylor instability (VOF, heavy fluid in blue)")
fig.tight_layout()

target_directory = r"D:\Dev\Dev Collage\Academic\Sem 7\CFD\LAB\Exp_4"
os.makedirs(target_directory, exist_ok=True)
output_file_path = os.path.join(target_directory, "experiment_4_rti_output.png")

fig.savefig(output_file_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Grid={grid_size_x}x{grid_size_y}; steps={total_time_steps}; time={current_time:.4f}")
print(f"Initial heavy-fluid area={initial_heavy_area:.12f}")
print(f"Final heavy-fluid area={final_heavy_area:.12f}")
print(f"Relative mass error={abs(final_heavy_area - initial_heavy_area)/initial_heavy_area:.3e}")
print(f"Volume fraction range=[{volume_fraction.min():.6f}, {volume_fraction.max():.6f}]")
print(f"Maximum projected divergence={maximum_divergence:.3e}")
print(f"Saved figure at: {output_file_path}")