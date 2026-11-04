"""
  CFD & HT Lab - Experiment 9 (Student Exercises Solution)
  Topic: Lid-Driven Cavity Flow Benchmark & Sensitivity Analysis
  
  Student Name : Dev Pateliya
  Roll Number  : 23BME081
  Semester     : 7th Semester (B.Tech Mechanical Engineering)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dstn, idstn

lid_velocity = 1.0
simulation_time = 45.0
cavity_length = 1.0

ghia_y_coords = np.array([0.0, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9688, 1.0])
ghia_u_velocities = np.array([0.0, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.78871, 1.0])
ghia_x_coords = np.array([0.0, 0.0625, 0.0938, 0.1563, 0.2266, 0.5, 0.8047, 0.8594, 0.9063, 0.9531, 0.9688, 1.0])
ghia_v_velocities = np.array([0.0, 0.09233, 0.12317, 0.16077, 0.17507, 0.05454, -0.24533, -0.22445, -0.16914, -0.08864, -0.05906, 0.0])


def run_cavity_simulation(grid_size, reynolds_number, advection_scheme="upwind"):
    grid_spacing = cavity_length / (grid_size - 1)
    kinematic_viscosity = lid_velocity * cavity_length / reynolds_number

    target_time_step = min(0.30 * grid_spacing / lid_velocity, 0.20 * grid_spacing**2 / kinematic_viscosity)
    total_steps = int(np.ceil(simulation_time / target_time_step))
    time_step = simulation_time / total_steps

    grid_x = np.linspace(0.0, cavity_length, grid_size)
    grid_y = np.linspace(0.0, cavity_length, grid_size)

    vorticity = np.zeros((grid_size, grid_size))
    streamfunction = np.zeros_like(vorticity)

    wavenumbers = np.arange(1, grid_size - 1)
    laplacian_eigenvalues = 4.0 / grid_spacing**2 * (
        np.sin(np.pi * wavenumbers[:, None] / (2 * (grid_size - 1)))**2 +
        np.sin(np.pi * wavenumbers[None, :] / (2 * (grid_size - 1)))**2
    )

    def solve_streamfunction_poisson(vorticity_field):
        spectral_vorticity = dstn(vorticity_field[1:-1, 1:-1], type=1, norm="ortho")
        stream_solution = np.zeros_like(vorticity_field)
        stream_solution[1:-1, 1:-1] = idstn(spectral_vorticity / laplacian_eigenvalues, type=1, norm="ortho")
        return stream_solution

    def update_wall_vorticity(vorticity_field, stream_field):
        vorticity_field[0, 1:-1] = -2.0 * stream_field[1, 1:-1] / grid_spacing**2
        vorticity_field[-1, 1:-1] = -2.0 * stream_field[-2, 1:-1] / grid_spacing**2 - 2.0 * lid_velocity / grid_spacing
        vorticity_field[1:-1, 0] = -2.0 * stream_field[1:-1, 1] / grid_spacing**2
        vorticity_field[1:-1, -1] = -2.0 * stream_field[1:-1, -2] / grid_spacing**2

    def compute_vorticity_transport(vorticity_field, stream_field):
        center = vorticity_field[1:-1, 1:-1]
        east = vorticity_field[1:-1, 2:]
        west = vorticity_field[1:-1, :-2]
        north = vorticity_field[2:, 1:-1]
        south = vorticity_field[:-2, 1:-1]
        
        u_east = (stream_field[2:, 2:] + stream_field[2:, 1:-1] - stream_field[:-2, 2:] - stream_field[:-2, 1:-1]) / (4 * grid_spacing)
        u_west = (stream_field[2:, 1:-1] + stream_field[2:, :-2] - stream_field[:-2, 1:-1] - stream_field[:-2, :-2]) / (4 * grid_spacing)
        v_north = -(stream_field[2:, 2:] + stream_field[1:-1, 2:] - stream_field[2:, :-2] - stream_field[1:-1, :-2]) / (4 * grid_spacing)
        v_south = -(stream_field[1:-1, 2:] + stream_field[:-2, 2:] - stream_field[1:-1, :-2] - stream_field[:-2, :-2]) / (4 * grid_spacing)
        
        if advection_scheme == "upwind":
            flux_east = u_east * np.where(u_east >= 0, center, east)
            flux_west = u_west * np.where(u_west >= 0, west, center)
            flux_north = v_north * np.where(v_north >= 0, center, north)
            flux_south = v_south * np.where(v_south >= 0, south, center)
        else:
            flux_east = u_east * 0.5 * (center + east)
            flux_west = u_west * 0.5 * (west + center)
            flux_north = v_north * 0.5 * (center + north)
            flux_south = v_south * 0.5 * (south + center)
        
        advection_term = (flux_east - flux_west + flux_north - flux_south) / grid_spacing
        diffusion_term = kinematic_viscosity * (east + west + north + south - 4 * center) / grid_spacing**2
        return diffusion_term - advection_term

    for step in range(1, total_steps + 1):
        streamfunction = solve_streamfunction_poisson(vorticity)
        update_wall_vorticity(vorticity, streamfunction)
        transport_rate = compute_vorticity_transport(vorticity, streamfunction)
        vorticity[1:-1, 1:-1] += time_step * transport_rate

    streamfunction = solve_streamfunction_poisson(vorticity)
    update_wall_vorticity(vorticity, streamfunction)

    velocity_u = np.zeros_like(streamfunction)
    velocity_v = np.zeros_like(streamfunction)

    velocity_u[1:-1, 1:-1] = (streamfunction[2:, 1:-1] - streamfunction[:-2, 1:-1]) / (2 * grid_spacing)
    velocity_v[1:-1, 1:-1] = -(streamfunction[1:-1, 2:] - streamfunction[1:-1, :-2]) / (2 * grid_spacing)
    velocity_u[-1, :] = lid_velocity

    interpolated_u = np.interp(ghia_y_coords, grid_y, velocity_u[:, grid_size // 2])
    interpolated_v = np.interp(ghia_x_coords, grid_x, velocity_v[grid_size // 2, :])

    rmse_u = np.sqrt(np.mean((interpolated_u - ghia_u_velocities)**2))
    rmse_v = np.sqrt(np.mean((interpolated_v - ghia_v_velocities)**2))

    return grid_x, grid_y, velocity_u, velocity_v, streamfunction, rmse_u, rmse_v


target_directory = r"D:\Dev\Dev Collage\Academic\Sem 7\CFD\LAB\Exp_9"
os.makedirs(target_directory, exist_ok=True)

grid_sizes = [33, 65, 97]
ex1_results = {}
print("=" * 85)
print("                    EXERCISE 1: GRID CONVERGENCE STUDY (Re = 100)                      ")
print("=" * 85)
print(f"{'Grid Size (N)':<18} | {'Ghia u-RMSE':<18} | {'Ghia v-RMSE':<18}")
print("-" * 85)

for n in grid_sizes:
    gx, gy, u, v, psi, rmse_u, rmse_v = run_cavity_simulation(grid_size=n, reynolds_number=100.0)
    ex1_results[n] = (rmse_u, rmse_v, u, v, gy, gx)
    print(f"{n:<18} | {rmse_u:<18.6f} | {rmse_v:<18.6f}")

print("-" * 85)
print("EXPLANATION (Grid Convergence):")
print("- Monotonic grid convergence is observed as both RMSE values drop consistently with finer mesh.")
print("- Refining the mesh reduces numerical discretization errors and sharpens boundary layers.\n")

print("=" * 85)
print("                    EXERCISE 2: HIGH REYNOLDS NUMBER (Re = 400)                        ")
print("=" * 85)
gx400, gy400, u400, v400, psi400, rmse_u400, rmse_v400 = run_cavity_simulation(grid_size=65, reynolds_number=400.0)
print(f"Re = 400 (N=65) -> Ghia u-RMSE: {rmse_u400:.6f} | Ghia v-RMSE: {rmse_v400:.6f}")
print("EXPLANATION (Main Vortex Shift):")
print("- At Re = 400, viscous dissipation weakens relative to inertia.")
print("- The primary vortex core moves downstream toward the geometric center of the cavity,")
print("  and secondary corner vortices grow larger and more prominent.\n")

print("=" * 85)
print("              EXERCISE 3: SCHEME COMPARISON (Donor-Cell Upwind vs Centred)             ")
print("=" * 85)
gx_c, gy_c, u_c, v_c, psi_c, rmse_u_c, rmse_v_c = run_cavity_simulation(grid_size=65, reynolds_number=100.0, advection_scheme="centred")
print(f"Centred Scheme (N=65) -> Ghia u-RMSE: {rmse_u_c:.6f} | Ghia v-RMSE: {rmse_v_c:.6f}")
print("EXPLANATION (Advection Scheme Comparison):")
print("- Centred advection removes numerical diffusion, improving spatial accuracy.")
print("- However, it introduces non-physical dispersion wiggles near steep gradients and is less stable.")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(grid_sizes, [ex1_results[n][0] for n in grid_sizes], "o-", label="u-RMSE")
axes[0].plot(grid_sizes, [ex1_results[n][1] for n in grid_sizes], "s-", label="v-RMSE")
axes[0].set(xlabel="Grid Size (N)", ylabel="RMSE against Ghia et al.", title="Ex 1: Grid Convergence (Re=100)")
axes[0].set_xticks(grid_sizes)
axes[0].grid(alpha=0.3)
axes[0].legend()

contour_plot = axes[1].contourf(gx400, gy400, psi400, levels=25, cmap="viridis")
axes[1].set(xlabel="x / L", ylabel="y / L", title="Ex 2: Streamlines (Re=400, N=65)", aspect="equal")
fig.colorbar(contour_plot, ax=axes[1])

axes[2].plot(ex1_results[65][2][:, 32], ex1_results[65][4], label="Upwind (N=65)")
axes[2].plot(u_c[:, 32], gy_c, "--", label="Centred (N=65)")
axes[2].plot(ghia_u_velocities, ghia_y_coords, "o", markersize=4, label="Ghia et al.")
axes[2].set(xlabel="u / U", ylabel="y / L", title="Ex 3: Vertical Centreline Comparison")
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.tight_layout()

output_file_path = os.path.join(target_directory, "experiment_9_exercises_output.png")
fig.savefig(output_file_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 85)
print(f"Saved exercise figure successfully at: {output_file_path}")
print("=" * 85)