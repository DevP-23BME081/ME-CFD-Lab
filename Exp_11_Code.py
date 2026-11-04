"""
  CFD & HT Lab - Experiment 11
  Topic: Generic Convection-Diffusion Transport
  
  Student Name : Dev Pateliya
  Roll Number  : 23BME081
  Semester     : 7th Semester (B.Tech Mechanical Engineering)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

num_cells_x = 75
num_cells_y = 35
prandtl_number = 0.71
peclet_numbers = [50, 200, 1000]

cases_config = {
    "Forced flat plate": {
        "domain_x": 1.0,
        "domain_y": 0.5,
        "initial_val": 0.0,
        "max_iterations": 4000
    },
    "Forced channel": {
        "domain_x": 4.0,
        "domain_y": 1.0,
        "initial_val": 0.0,
        "max_iterations": 4000
    },
    "Natural-convection surrogate": {
        "domain_x": 1.0,
        "domain_y": 1.0,
        "initial_val": 0.5,
        "max_iterations": 4000
    }
}


def compute_velocity_fields(case_name, peclet_val, domain_length_x, domain_length_y):
    reference_velocity = 1.0
    reference_length = domain_length_y if case_name != "Forced flat plate" else domain_length_x
    diffusivity = reference_velocity * reference_length / peclet_val

    cell_width_x = domain_length_x / num_cells_x
    cell_width_y = domain_length_y / num_cells_y

    face_x_coords = np.linspace(0.0, domain_length_x, num_cells_x + 1)
    face_y_coords = np.linspace(0.0, domain_length_y, num_cells_y + 1)
    cell_x_coords = 0.5 * (face_x_coords[:-1] + face_x_coords[1:])
    cell_y_coords = 0.5 * (face_y_coords[:-1] + face_y_coords[1:])

    u_east_face = np.zeros((num_cells_y, num_cells_x + 1))
    v_north_face = np.zeros((num_cells_y + 1, num_cells_x))

    if case_name == "Forced flat plate":
        kinematic_viscosity = diffusivity / prandtl_number
        for index_y in range(num_cells_y):
            for index_x in range(num_cells_x + 1):
                position_x = max(face_x_coords[index_x], 1.0e-5)
                position_y = cell_y_coords[index_y]
                boundary_layer_thickness = 4.64 * np.sqrt(kinematic_viscosity * position_x / reference_velocity)
                eta_val = min(position_y / boundary_layer_thickness, 1.0)
                u_east_face[index_y, index_x] = reference_velocity * (1.5 * eta_val - 0.5 * eta_val**3)

        for index_y in range(num_cells_y + 1):
            for index_x in range(num_cells_x):
                position_x = max(cell_x_coords[index_x], 1.0e-5)
                position_y = face_y_coords[index_y]
                boundary_layer_thickness = 4.64 * np.sqrt(kinematic_viscosity * position_x / reference_velocity)
                eta_val = min(position_y / boundary_layer_thickness, 1.0)
                v_north_face[index_y, index_x] = reference_velocity * (0.75 * eta_val**2 - 0.125 * eta_val**4) * (boundary_layer_thickness / position_x)

    elif case_name == "Forced channel":
        for index_y in range(num_cells_y):
            position_y = cell_y_coords[index_y]
            velocity_profile = 1.5 * reference_velocity * (1.0 - (2.0 * position_y / domain_length_y - 1.0)**2)
            u_east_face[index_y, :] = velocity_profile
        v_north_face.fill(0.0)

    elif case_name == "Natural-convection surrogate":
        stream_function_faces = np.zeros((num_cells_y + 1, num_cells_x + 1))
        for index_y in range(num_cells_y + 1):
            for index_x in range(num_cells_x + 1):
                pos_x = face_x_coords[index_x]
                pos_y = face_y_coords[index_y]
                stream_function_faces[index_y, index_x] = -(reference_velocity * domain_length_y / np.pi) * (np.sin(np.pi * pos_x / domain_length_x)**2) * (np.sin(np.pi * pos_y / domain_length_y)**2)

        u_east_face = (stream_function_faces[1:, :] - stream_function_faces[:-1, :]) / cell_width_y
        v_north_face = -(stream_function_faces[:, 1:] - stream_function_faces[:, :-1]) / cell_width_x

    return u_east_face, v_north_face, diffusivity, cell_width_x, cell_width_y, cell_x_coords, cell_y_coords


def run_convection_diffusion_solver(case_name, peclet_val):
    start_cpu_time = time.time()
    config = cases_config[case_name]
    domain_length_x = config["domain_x"]
    domain_length_y = config["domain_y"]
    initial_value = config["initial_val"]
    max_iterations = config["max_iterations"]

    u_east_face, v_north_face, diffusivity, cell_width_x, cell_width_y, cell_x_coords, cell_y_coords = compute_velocity_fields(
        case_name, peclet_val, domain_length_x, domain_length_y
    )

    u_west_face = u_east_face[:, :-1]
    u_east_face = u_east_face[:, 1:]
    v_south_face = v_north_face[:-1, :]
    v_north_face = v_north_face[1:, :]

    u_cell_center = 0.5 * (u_east_face + u_west_face)
    v_cell_center = 0.5 * (v_north_face + v_south_face)
    velocity_magnitude = np.sqrt(u_cell_center**2 + v_cell_center**2)
    max_cell_peclet = float(np.max(velocity_magnitude * cell_width_x / diffusivity))

    max_u = np.max(np.abs(u_cell_center))
    max_v = np.max(np.abs(v_cell_center))
    time_step = 0.8 / (max_u / cell_width_x + max_v / cell_width_y + 2.0 * diffusivity * (1.0 / cell_width_x**2 + 1.0 / cell_width_y**2))

    scalar_field = np.full((num_cells_y, num_cells_x), initial_value)
    initial_residual_norm = None
    converged_status = False
    iterations_performed = 0
    residual_history = []

    for iteration in range(1, max_iterations + 1):
        scalar_east = np.zeros((num_cells_y, num_cells_x))
        scalar_west = np.zeros((num_cells_y, num_cells_x))
        scalar_north = np.zeros((num_cells_y, num_cells_x))
        scalar_south = np.zeros((num_cells_y, num_cells_x))

        scalar_east[:, :-1] = scalar_field[:, 1:]
        scalar_east[:, -1] = scalar_field[:, -1]

        scalar_west[:, 1:] = scalar_field[:, :-1]
        if case_name in ["Forced flat plate", "Forced channel"]:
            scalar_west[:, 0] = 0.0
        else:
            scalar_west[:, 0] = 1.0

        scalar_north[:-1, :] = scalar_field[1:, :]
        if case_name == "Forced flat plate":
            scalar_north[-1, :] = 0.0
        elif case_name == "Forced channel":
            scalar_north[-1, :] = 1.0
        else:
            scalar_north[-1, :] = scalar_field[-1, :]

        scalar_south[1:, :] = scalar_field[:-1, :]
        if case_name == "Forced flat plate":
            is_heated_plate = (cell_x_coords >= 0.1 * domain_length_x)
            scalar_south[0, is_heated_plate] = 1.0
            scalar_south[0, ~is_heated_plate] = scalar_field[0, ~is_heated_plate]
        elif case_name == "Forced channel":
            scalar_south[0, :] = 1.0
        else:
            scalar_south[0, :] = scalar_field[0, :]

        if case_name == "Natural-convection surrogate":
            scalar_east[:, -1] = 0.0

        convective_flux_east = np.maximum(u_east_face, 0.0) * scalar_field + np.minimum(u_east_face, 0.0) * scalar_east
        convective_flux_west = np.maximum(u_west_face, 0.0) * scalar_west + np.minimum(u_west_face, 0.0) * scalar_field
        convective_flux_north = np.maximum(v_north_face, 0.0) * scalar_field + np.minimum(v_north_face, 0.0) * scalar_north
        convective_flux_south = np.maximum(v_south_face, 0.0) * scalar_south + np.minimum(v_south_face, 0.0) * scalar_field

        diffusive_grad_east = (scalar_field[:, 1:] - scalar_field[:, :-1]) / cell_width_x
        diffusive_grad_east_full = np.zeros((num_cells_y, num_cells_x))
        diffusive_grad_east_full[:, :-1] = diffusive_grad_east
        if case_name == "Natural-convection surrogate":
            diffusive_grad_east_full[:, -1] = (0.0 - scalar_field[:, -1]) / (0.5 * cell_width_x)
        else:
            diffusive_grad_east_full[:, -1] = 0.0

        diffusive_grad_west_full = np.zeros((num_cells_y, num_cells_x))
        diffusive_grad_west_full[:, 1:] = diffusive_grad_east
        if case_name in ["Forced flat plate", "Forced channel"]:
            diffusive_grad_west_full[:, 0] = (scalar_field[:, 0] - 0.0) / (0.5 * cell_width_x)
        else:
            diffusive_grad_west_full[:, 0] = (scalar_field[:, 0] - 1.0) / (0.5 * cell_width_x)

        diffusive_grad_north = (scalar_field[1:, :] - scalar_field[:-1, :]) / cell_width_y
        diffusive_grad_north_full = np.zeros((num_cells_y, num_cells_x))
        diffusive_grad_north_full[:-1, :] = diffusive_grad_north
        if case_name == "Forced flat plate":
            diffusive_grad_north_full[-1, :] = (0.0 - scalar_field[-1, :]) / (0.5 * cell_width_y)
        elif case_name == "Forced channel":
            diffusive_grad_north_full[-1, :] = (1.0 - scalar_field[-1, :]) / (0.5 * cell_width_y)
        else:
            diffusive_grad_north_full[-1, :] = 0.0

        diffusive_grad_south_full = np.zeros((num_cells_y, num_cells_x))
        diffusive_grad_south_full[1:, :] = diffusive_grad_north
        if case_name == "Forced flat plate":
            is_heated_plate = (cell_x_coords >= 0.1 * domain_length_x)
            diffusive_grad_south_full[0, is_heated_plate] = (scalar_field[0, is_heated_plate] - 1.0) / (0.5 * cell_width_y)
            diffusive_grad_south_full[0, ~is_heated_plate] = 0.0
        elif case_name == "Forced channel":
            diffusive_grad_south_full[0, :] = (scalar_field[0, :] - 1.0) / (0.5 * cell_width_y)
        else:
            diffusive_grad_south_full[0, :] = 0.0

        rhs_terms = (
            -(convective_flux_east - convective_flux_west) / cell_width_x
            -(convective_flux_north - convective_flux_south) / cell_width_y
            + diffusivity * (diffusive_grad_east_full - diffusive_grad_west_full) / cell_width_x
            + diffusivity * (diffusive_grad_north_full - diffusive_grad_south_full) / cell_width_y
        )

        current_residual_norm = np.sqrt(np.mean(rhs_terms**2))
        if initial_residual_norm is None or initial_residual_norm == 0.0:
            initial_residual_norm = max(current_residual_norm, 1.0e-12)

        relative_residual = current_residual_norm / initial_residual_norm
        residual_history.append(relative_residual)
        scalar_field += time_step * rhs_terms
        iterations_performed = iteration

        if relative_residual < 1.0e-6:
            converged_status = True
            break

    elapsed_cpu_time = time.time() - start_cpu_time

    if case_name == "Forced flat plate":
        is_heated_plate = (cell_x_coords >= 0.1 * domain_length_x)
        wall_flux = diffusivity * (1.0 - scalar_field[0, is_heated_plate]) / (0.5 * cell_width_y)
        local_nusselt = wall_flux * domain_length_x / diffusivity
        nusselt_average = float(np.mean(local_nusselt))
        outlet_scalar_mean = float(np.mean(scalar_field[:, -1]))
        nusselt_coord = cell_x_coords[is_heated_plate]

    elif case_name == "Forced channel":
        wall_flux_south = diffusivity * (1.0 - scalar_field[0, :]) / (0.5 * cell_width_y)
        wall_flux_north = diffusivity * (1.0 - scalar_field[-1, :]) / (0.5 * cell_width_y)
        local_nusselt = 0.5 * (wall_flux_south + wall_flux_north) * domain_length_y / diffusivity
        nusselt_average = float(np.mean(local_nusselt))
        bulk_velocity_outlet = u_east_face[:, -1]
        outlet_scalar_mean = float(np.sum(bulk_velocity_outlet * scalar_field[:, -1]) / np.sum(bulk_velocity_outlet))
        nusselt_coord = cell_x_coords

    elif case_name == "Natural-convection surrogate":
        wall_flux_hot = diffusivity * (1.0 - scalar_field[:, 0]) / (0.5 * cell_width_x)
        local_nusselt = wall_flux_hot * domain_length_y / diffusivity
        nusselt_average = float(np.mean(local_nusselt))
        outlet_scalar_mean = float(np.mean(scalar_field))
        nusselt_coord = cell_y_coords

    return scalar_field, nusselt_average, outlet_scalar_mean, max_cell_peclet, iterations_performed, converged_status, cell_x_coords, cell_y_coords, residual_history, elapsed_cpu_time, local_nusselt, nusselt_coord


target_directory = r"D:\Dev\Dev Collage\Academic\Sem 7\CFD\LAB\11"
os.makedirs(target_directory, exist_ok=True)

simulation_results = []
contour_fields_pe200 = {}

for case_name in cases_config.keys():
    for peclet_val in peclet_numbers:
        (scalar_field, nu_avg, outlet_val, max_pe_cell, iters, 
         converged, cell_x, cell_y, res_hist, cpu_time, loc_nu, nu_coord) = run_convection_diffusion_solver(case_name, peclet_val)
        
        simulation_results.append({
            "case": case_name,
            "pe": peclet_val,
            "iters": iters,
            "converged": converged,
            "nu": nu_avg,
            "outlet": outlet_val,
            "max_pe_cell": max_pe_cell,
            "cpu_time": cpu_time,
            "res_hist": res_hist,
            "loc_nu": loc_nu,
            "nu_coord": nu_coord
        })
        if peclet_val == 200:
            contour_fields_pe200[case_name] = (scalar_field, cell_x, cell_y, res_hist, loc_nu, nu_coord)

fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))

case_key = "Forced flat plate"
field_fp, cx_fp, cy_fp, res_fp, nu_fp, coord_fp = contour_fields_pe200[case_key]
im0 = axes1[0, 0].imshow(field_fp, origin="lower", extent=[0, 1.0, 0, 0.5], cmap="magma", vmin=0, vmax=1)
axes1[0, 0].set_title(f"{case_key} (Pe=200)")
axes1[0, 0].set_xlabel("x")
axes1[0, 0].set_ylabel("y")
fig1.colorbar(im0, ax=axes1[0, 0])

case_key = "Forced channel"
field_fc, cx_fc, cy_fc, res_fc, nu_fc, coord_fc = contour_fields_pe200[case_key]
im1 = axes1[0, 1].imshow(field_fc, origin="lower", extent=[0, 4.0, 0, 1.0], cmap="magma", vmin=0, vmax=1)
axes1[0, 1].set_title(f"{case_key} (Pe=200)")
axes1[0, 1].set_xlabel("x")
axes1[0, 1].set_ylabel("y")
fig1.colorbar(im1, ax=axes1[0, 1])

case_key = "Natural-convection surrogate"
field_nc, cx_nc, cy_nc, res_nc, nu_nc, coord_nc = contour_fields_pe200[case_key]
im2 = axes1[1, 0].imshow(field_nc, origin="lower", extent=[0, 1.0, 0, 1.0], cmap="magma", vmin=0, vmax=1)
axes1[1, 0].set_title(f"{case_key} (Pe=200)")
axes1[1, 0].set_xlabel("x")
axes1[1, 0].set_ylabel("y")
fig1.colorbar(im2, ax=axes1[1, 0])

axes1[1, 1].plot(coord_fp, nu_fp, label="Flat Plate Nu(x)", color="#1f77b4")
axes1[1, 1].plot(coord_fc, nu_fc, label="Channel Nu(x)", color="#ff7f0e")
axes1[1, 1].plot(coord_nc, nu_nc, label="Cavity Hot-Wall Nu(y)", color="#2ca02c")
axes1[1, 1].set_xlabel("Wall Coordinate (x or y)")
axes1[1, 1].set_ylabel("Local Nusselt Number Nu_local")
axes1[1, 1].set_title("Local Nusselt Number Profiles (Pe=200)")
axes1[1, 1].grid(alpha=0.3)
axes1[1, 1].legend()

fig1.tight_layout()
contour_output_path = os.path.join(target_directory, "experiment_11_comprehensive_diagnostics.png")
fig1.savefig(contour_output_path, dpi=180, bbox_inches="tight")
plt.close(fig1)

fig2, axis2 = plt.subplots(figsize=(7, 5))
for case_name in cases_config.keys():
    pe_vals = [res["pe"] for res in simulation_results if res["case"] == case_name]
    nu_vals = [res["nu"] for res in simulation_results if res["case"] == case_name]
    axis2.loglog(pe_vals, nu_vals, "o-", label=case_name, linewidth=1.8, markersize=6)

axis2.set_xlabel("Pe")
axis2.set_ylabel("Average Nu")
axis2.set_title("Nusselt number vs Peclet number")
axis2.grid(True, which="both", linestyle="--", alpha=0.4)
axis2.legend()
fig2.tight_layout()
nusselt_output_path = os.path.join(target_directory, "experiment_11_nusselt_vs_peclet.png")
fig2.savefig(nusselt_output_path, dpi=160, bbox_inches="tight")
plt.close(fig2)

print("=" * 110)
print("                     CFD LAB - EXPERIMENT 11 EXTENDED RESULTS TABLE                  ")
print("=" * 110)
print(f"{'Case':<30} | {'Pe':<6} | {'Iters':<8} | {'Status':<10} | {'Nu_avg':<9} | {'Outlet phi':<12} | {'Max Pe_cell':<12} | {'CPU Time (s)':<12}")
print("-" * 110)

for res in simulation_results:
    status_str = "Converged" if res["converged"] else "Max Iters"
    print(
        f"{res['case']:<30} | {res['pe']:<6} | {res['iters']:<8} | {status_str:<10} | "
        f"{res['nu']:<9.4f} | {res['outlet']:<12.4f} | {res['max_pe_cell']:<12.3f} | {res['cpu_time']:<12.3f}"
    )

print("=" * 110)
print(f"Saved extended diagnostics figure at: {contour_output_path}")
print(f"Saved Nu vs Pe scaling plot at:       {nusselt_output_path}")
print("=" * 110)