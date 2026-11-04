"""
  CFD & HT Lab - Experiment 10 (Student Exercises Solution)
  Topic: Turbulent Pipe-Flow Model Comparison & Sensitivity Analysis
  
  Student Name : Dev Pateliya
  Roll Number  : 23BME081
  Semester     : 7th Semester (B.Tech Mechanical Engineering)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

pipe_radius = 0.025
kinematic_viscosity = 1.5e-5
fluid_density = 1.2
karman_constant = 0.41
max_iterations = 1000
tolerance = 1.0e-10


def solve_flow_matrix(effective_viscosity, num_cells, source_term, cell_volumes, face_areas, grid_spacing):
    face_conductance = face_areas[1:-1] * effective_viscosity[1:-1] / grid_spacing
    main_diagonal = np.zeros(num_cells)
    main_diagonal[:-1] += face_conductance
    main_diagonal[1:] += face_conductance
    
    wall_conductance = face_areas[-1] * kinematic_viscosity / (0.5 * grid_spacing)
    main_diagonal[-1] += wall_conductance
    
    system_matrix = diags([-face_conductance, main_diagonal, -face_conductance], [-1, 0, 1], format="csr")
    return spsolve(system_matrix, source_term * cell_volumes)


def run_single_simulation(model_name, num_cells, friction_velocity, damping_aplus=26.0):
    grid_spacing = pipe_radius / num_cells
    cell_centers = (np.arange(num_cells) + 0.5) * grid_spacing
    cell_faces = np.arange(num_cells + 1) * grid_spacing
    
    source_term = 2.0 * friction_velocity**2 / pipe_radius
    cell_volumes = np.pi * (cell_faces[1:]**2 - cell_faces[:-1]**2)
    face_areas = 2.0 * np.pi * cell_faces

    velocity_profile = solve_flow_matrix(np.full(num_cells + 1, kinematic_viscosity), num_cells, source_term, cell_volumes, face_areas, grid_spacing)
    
    for iteration in range(1, max_iterations + 1):
        velocity_gradient = np.zeros(num_cells + 1)
        velocity_gradient[1:-1] = (velocity_profile[1:] - velocity_profile[:-1]) / grid_spacing
        velocity_gradient[-1] = -velocity_profile[-1] / (0.5 * grid_spacing)
        
        distance_from_wall = pipe_radius - cell_faces
        mixing_length = karman_constant * distance_from_wall
        
        if model_name == "van Driest":
            damping_factor = 1.0 - np.exp(-distance_from_wall * friction_velocity / (damping_aplus * kinematic_viscosity))
            mixing_length = mixing_length * damping_factor
            
        effective_viscosity = kinematic_viscosity + (mixing_length**2) * np.abs(velocity_gradient)
        effective_viscosity[-1] = kinematic_viscosity
        
        updated_velocity = solve_flow_matrix(effective_viscosity, num_cells, source_term, cell_volumes, face_areas, grid_spacing)
        convergence_error = float(np.max(np.abs(updated_velocity - velocity_profile)))
        
        velocity_profile = 0.25 * updated_velocity + 0.75 * velocity_profile
        
        if convergence_error < tolerance:
            velocity_profile = updated_velocity
            break
            
    bulk_velocity = float(np.sum(cell_volumes * velocity_profile) / (np.pi * pipe_radius**2))
    reynolds_number = 2.0 * pipe_radius * bulk_velocity / kinematic_viscosity
    darcy_friction_factor = 8.0 * (friction_velocity / bulk_velocity)**2
    
    return bulk_velocity, reynolds_number, darcy_friction_factor


target_directory = r"D:\Dev\Dev Collage\Academic\Sem 7\CFD\LAB\Exp_10"
os.makedirs(target_directory, exist_ok=True)

mesh_sizes = [60, 120, 240]
results_ex1 = {"Prandtl": {"ub": [], "fd": [], "re": []}, "van Driest": {"ub": [], "fd": [], "re": []}}

for n_cells in mesh_sizes:
    for model in ["Prandtl", "van Driest"]:
        ub, re_d, f_d = run_single_simulation(model, n_cells, friction_velocity=0.18)
        results_ex1[model]["ub"].append(ub)
        results_ex1[model]["fd"].append(f_d)
        results_ex1[model]["re"].append(re_d)

aplus_values = [20, 23, 26, 29, 32]
results_ex2 = {"ub": [], "fd": [], "re": []}
for aplus in aplus_values:
    ub, re_d, f_d = run_single_simulation("van Driest", 120, friction_velocity=0.18, damping_aplus=aplus)
    results_ex2["ub"].append(ub)
    results_ex2["fd"].append(f_d)
    results_ex2["re"].append(re_d)

results_ex3 = {}
for model in ["Prandtl", "van Driest"]:
    ub, re_d, f_d = run_single_simulation(model, 120, friction_velocity=0.25)
    blasius_f = 0.3164 * (re_d**(-0.25))
    pct_dev = ((f_d - blasius_f) / blasius_f) * 100.0
    results_ex3[model] = {"ub": ub, "re": re_d, "fd": f_d, "blasius": blasius_f, "dev": pct_dev}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(mesh_sizes, results_ex1["Prandtl"]["fd"], "o-", label="Prandtl", linewidth=1.8)
axes[0].plot(mesh_sizes, results_ex1["van Driest"]["fd"], "s-", label="van Driest", linewidth=1.8)
axes[0].set_xlabel("Number of Radial Cells (N)")
axes[0].set_ylabel("Darcy Friction Factor f_D")
axes[0].set_title("Ex 1: Mesh Sensitivity Analysis")
axes[0].set_xticks(mesh_sizes)
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(aplus_values, results_ex2["fd"], "o-", color="#d62728", linewidth=1.8)
axes[1].set_xlabel("van Driest Constant A+")
axes[1].set_ylabel("Darcy Friction Factor f_D")
axes[1].set_title("Ex 2: Near-Wall Damping Sensitivity")
axes[1].set_xticks(aplus_values)
axes[1].grid(alpha=0.3)

models_ex3 = ["Prandtl", "van Driest"]
fd_ex3 = [results_ex3[m]["fd"] for m in models_ex3]
blasius_ex3 = [results_ex3[m]["blasius"] for m in models_ex3]
x_indices = np.arange(len(models_ex3))
bar_width = 0.35

axes[2].bar(x_indices - bar_width/2, fd_ex3, bar_width, label="Model f_D", color="#1f77b4")
axes[2].bar(x_indices + bar_width/2, blasius_ex3, bar_width, label="Blasius f_D", color="#ff7f0e")
axes[2].set_xticks(x_indices)
axes[2].set_xticklabels(models_ex3)
axes[2].set_ylabel("Darcy Friction Factor f_D")
axes[2].set_title("Ex 3: High Re (u_tau = 0.25 m/s)")
axes[2].grid(alpha=0.3, axis="y")
axes[2].legend()

fig.tight_layout()

output_file_path = os.path.join(target_directory, "experiment_10_exercises_all_plots.png")
fig.savefig(output_file_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print("=" * 85)
print("                       EXERCISE 1: MESH SENSITIVITY ANALYSIS                          ")
print("=" * 85)
print(f"{'Radial Cells (N)':<18} | {'Model':<12} | {'Ub (m/s)':<12} | {'Darcy f_D':<12} | {'Re_D':<10}")
print("-" * 85)
for i, n_cells in enumerate(mesh_sizes):
    for model in ["Prandtl", "van Driest"]:
        print(f"{n_cells:<18} | {model:<12} | {results_ex1[model]['ub'][i]:<12.6f} | {results_ex1[model]['fd'][i]:<12.6f} | {results_ex1[model]['re'][i]:<10.1f}")

print("\n" + "=" * 85)
print("                  EXERCISE 2: VAN DRIEST A+ PARAMETER SENSITIVITY                     ")
print("=" * 85)
print(f"{'A+ Parameter':<18} | {'Ub (m/s)':<12} | {'Darcy f_D':<12} | {'Re_D':<10}")
print("-" * 85)
for i, aplus in enumerate(aplus_values):
    print(f"{aplus:<18} | {results_ex2['ub'][i]:<12.6f} | {results_ex2['fd'][i]:<12.6f} | {results_ex2['re'][i]:<10.1f}")

print("\n" + "=" * 85)
print("                  EXERCISE 3: HIGH REYNOLDS NUMBER (u_tau = 0.25 m/s)                 ")
print("=" * 85)
print(f"{'Model Name':<15} | {'Re_D':<10} | {'Model f_D':<12} | {'Blasius f_D':<12} | {'% Deviation':<12}")
print("-" * 85)
for model in ["Prandtl", "van Driest"]:
    res = results_ex3[model]
    print(f"{model:<15} | {res['re']:<10.1f} | {res['fd']:<12.6f} | {res['blasius']:<12.6f} | {res['dev']:<+12.2f}%")

print("=" * 85)
print(f"Combined exercise figure saved at: {output_file_path}")
print("=" * 85)