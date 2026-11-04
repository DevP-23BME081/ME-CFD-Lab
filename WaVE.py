"""
  CFD & HT Lab - Experiment 3 (Student Exercises Solution)
  Topic: 1D Wave Propagation in a Still Lake (FVM Sensitivity & Stability)
  
  Student Name : Dev Pateliya
  Roll Number  : 23BME081
  Semester     : 7th Semester (B.Tech Mechanical Engineering)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized


def run_wave_simulation(num_cells=200, target_cfl=0.8, allow_instability=False):
    channel_length = 10.0
    water_depth = 1.0
    gravity = 9.81
    initial_amplitude = 0.10
    gaussian_width = 0.40
    total_simulation_time = 0.80

    grid_spacing = channel_length / num_cells
    wave_speed = np.sqrt(gravity * water_depth)

    if allow_instability:
        time_step = target_cfl * grid_spacing / wave_speed
        total_steps = int(np.ceil(total_simulation_time / time_step))
        time_step = total_simulation_time / total_steps
        courant_number = wave_speed * time_step / grid_spacing
    else:
        total_steps = int(np.ceil(total_simulation_time * wave_speed / (target_cfl * grid_spacing)))
        time_step = total_simulation_time / total_steps
        courant_number = wave_speed * time_step / grid_spacing

    grid_center_x = (np.arange(num_cells) + 0.5) * grid_spacing
    initial_surface = initial_amplitude * np.exp(-((grid_center_x - channel_length / 2) / gaussian_width) ** 2)

    right_wave_explicit = initial_surface.copy()
    left_wave_explicit = initial_surface.copy()
    right_wave_implicit = initial_surface.copy()
    left_wave_implicit = initial_surface.copy()

    matrix_right = lil_matrix((num_cells, num_cells))
    matrix_left = lil_matrix((num_cells, num_cells))

    for cell_index in range(num_cells):
        matrix_right[cell_index, cell_index] = 1 + courant_number
        matrix_right[cell_index, (cell_index - 1) % num_cells] = -courant_number

        matrix_left[cell_index, cell_index] = 1 + courant_number
        matrix_left[cell_index, (cell_index + 1) % num_cells] = -courant_number

    solve_right_matrix = factorized(matrix_right.tocsc())
    solve_left_matrix = factorized(matrix_left.tocsc())

    for _ in range(total_steps):
        right_wave_explicit = right_wave_explicit - courant_number * (right_wave_explicit - np.roll(right_wave_explicit, 1))
        left_wave_explicit = left_wave_explicit - courant_number * (left_wave_explicit - np.roll(left_wave_explicit, -1))

        right_wave_implicit = solve_right_matrix(right_wave_implicit)
        left_wave_implicit = solve_left_matrix(left_wave_implicit)

    explicit_surface = 0.5 * (right_wave_explicit + left_wave_explicit)
    implicit_surface = 0.5 * (right_wave_implicit + left_wave_implicit)

    exact_surface = 0.5 * initial_amplitude * (
        np.exp(-((grid_center_x - channel_length / 2 - wave_speed * total_simulation_time) / gaussian_width) ** 2)
        + np.exp(-((grid_center_x - channel_length / 2 + wave_speed * total_simulation_time) / gaussian_width) ** 2)
    )

    explicit_l2_error = np.sqrt(grid_spacing * np.sum((explicit_surface - exact_surface) ** 2))
    implicit_l2_error = np.sqrt(grid_spacing * np.sum((implicit_surface - exact_surface) ** 2))

    return grid_center_x, exact_surface, explicit_surface, implicit_surface, explicit_l2_error, implicit_l2_error, courant_number, time_step


target_directory = r"D:\Dev\Dev Collage\Academic\Sem 7\CFD\LAB\Wave_Propagation_FVM"
os.makedirs(target_directory, exist_ok=True)

print("=" * 85)
print("              EXERCISE 1: MESH REFINEMENT STUDY (N = 100, 200, 400)                   ")
print("=" * 85)
print(f"{'Cells (N)':<12} | {'CFL':<10} | {'Explicit L2 Error (E2)':<24} | {'Implicit L2 Error (E2)':<24}")
print("-" * 85)

ex1_mesh_sizes = [100, 200, 400]
ex1_results = {}

for n in ex1_mesh_sizes:
    x, exact, exp_surf, imp_surf, e2_exp, e2_imp, cfl, dt = run_wave_simulation(num_cells=n, target_cfl=0.8)
    ex1_results[n] = (e2_exp, e2_imp)
    print(f"{n:<12} | {cfl:<10.4f} | {e2_exp:<24.8f} | {e2_imp:<24.8f}")

print("\nEXPLANATION (Mesh Refinement):")
print("- As N increases from 100 to 400, grid cell spacing dx shrinks, reducing numerical diffusion.")
print("- Both explicit and implicit L2 errors (E2) drop significantly, confirming spatial convergence.\n")

print("=" * 85)
print("               EXERCISE 2: CFL REDUCTION STUDY (CFL = 0.8 vs 0.4)                      ")
print("=" * 85)
print(f"{'CFL Target':<12} | {'Time Step dt (s)':<18} | {'Explicit E2':<18} | {'Implicit E2':<18}")
print("-" * 85)

ex2_cfl_targets = [0.8, 0.4]
ex2_data = {}

for cfl_val in ex2_cfl_targets:
    x, exact, exp_surf, imp_surf, e2_exp, e2_imp, cfl_act, dt = run_wave_simulation(num_cells=200, target_cfl=cfl_val)
    ex2_data[cfl_val] = (x, exact, exp_surf, imp_surf, e2_exp, e2_imp)
    print(f"{cfl_val:<12.1f} | {dt:<18.7f} | {e2_exp:<18.8f} | {e2_imp:<18.8f}")

print("\nEXPLANATION (Implicit Sharpness):")
print("- Reducing CFL from 0.8 to 0.4 decreases the time step dt.")
print("- For the implicit backward-Euler solver, a smaller dt significantly reduces temporal numerical damping,")
print("  making the implicit wave peak sharper and bringing its profile closer to the exact solution.\n")

print("=" * 85)
print("               EXERCISE 3: EXPLICIT STABILITY CHECK (CFL > 1.0)                        ")
print("=" * 85)
x_stab, exact_stab, exp_stab, imp_stab, e2_exp_s, e2_imp_s, cfl_stab, dt_stab = run_wave_simulation(num_cells=200, target_cfl=1.1, allow_instability=True)
print(f"Tested Explicit CFL = {cfl_stab:.4f} (> 1.0)")
print(f"Explicit Max Elevation: {np.max(exp_stab):.4e} m (Unbounded Blow-up)")
print(f"Implicit Max Elevation: {np.max(imp_stab):.4f} m (Unconditionally Stable)")

print("\nEXPLANATION (Instability):")
print("- When explicit CFL exceeds 1.0, information travels faster than 1 mesh cell per time step.")
print("- The explicit forward-Euler scheme violates the CFL domain-of-dependence condition, causing severe")
print("  non-physical numerical oscillations and unbounded exponential blow-up.")
print("- The implicit backward-Euler scheme remains stable and bounded even when CFL > 1.0.\n")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(ex1_mesh_sizes, [ex1_results[n][0] for n in ex1_mesh_sizes], "o-", label="Explicit E2", color="#126782")
axes[0].plot(ex1_mesh_sizes, [ex1_results[n][1] for n in ex1_mesh_sizes], "s-", label="Implicit E2", color="#ae6033")
axes[0].set(xlabel="Grid Cells (N)", ylabel="L2 Error (E2)", title="Ex 1: Mesh Convergence (CFL=0.8)")
axes[0].set_xticks(ex1_mesh_sizes)
axes[0].grid(alpha=0.3)
axes[0].legend()

x_ex2, exact_ex2, exp_08, imp_08, _, _ = ex2_data[0.8]
_, _, _, imp_04, _, _ = ex2_data[0.4]
axes[1].plot(x_ex2, exact_ex2, "k-", label="Exact Solution", lw=1.5)
axes[1].plot(x_ex2, imp_08, color="#ae6033", linestyle="--", label="Implicit (CFL=0.8)")
axes[1].plot(x_ex2, imp_04, color="#2a9d8f", linestyle="-", label="Implicit (CFL=0.4)")
axes[1].set(xlabel="x (m)", ylabel="Eta (m)", title="Ex 2: Implicit Damping vs CFL")
axes[1].grid(alpha=0.3)
axes[1].legend()

axes[2].plot(x_stab, exact_stab, "k-", label="Exact Solution")
axes[2].plot(x_stab, exp_stab, color="#e63946", label="Explicit (CFL=1.1, Blow-up)")
axes[2].plot(x_stab, imp_stab, color="#ae6033", label="Implicit (CFL=1.1, Stable)")
axes[2].set(xlabel="x (m)", ylabel="Eta (m)", title="Ex 3: Instability Test (CFL=1.1)")
axes[2].set_ylim(-0.2, 0.4)
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.tight_layout()

output_file_path = os.path.join(target_directory, "experiment_3_exercises_output.png")
fig.savefig(output_file_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print("=" * 85)
print("         EXERCISE 4: PERIODIC BOUNDARY VS REFLECTIVE LAKE WALL EXPLANATION             ")
print("=" * 85)
print("1. Periodic Computational Boundary:")
print("   - Connects the left and right domain ends mathematically: node i = 0 links to node i = N-1.")
print("   - Waves reaching the right boundary pass through and re-enter seamlessly from the left side.")
print("   - Mass and momentum flux exit one end and enter the opposite end without reversing direction.")

print("\n2. Reflective Solid Wall (Physical Lake Boundary):")
print("   - Imposes a solid wall boundary condition (zero normal velocity, u = 0).")
print("   - Incoming waves bounce back into the domain with inverted velocity and non-zero pressure buildup.")
print("   - A periodic boundary is used only to isolate wave propagation from wall reflections before")
print("     waves hit the domain ends.")
print("=" * 85)
print(f"Saved exercise figure at: {output_file_path}")
print("=" * 85)