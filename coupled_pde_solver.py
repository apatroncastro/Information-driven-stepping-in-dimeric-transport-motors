"""
Solver for coupled PDEs with delta function source terms:
∂_t P_0 = ∂_r ((r+f)P_0 + ∂_r P_0) - k_0*(e^{Δμ/2}δ(r-a) + e^{-Δμ/2}δ(r+a))P_0 + k_0*e^{-Δμ/2}δ(r-a)P_1
∂_t P_1 = ∂_r ((r-f)P_1 + ∂_r P_1) - k_0*(e^{Δμ/2}δ(r+a) + e^{-Δμ/2}δ(r-a))P_1 + k_0*e^{Δμ/2}δ(r-a)P_0

Initial conditions: P_0(r,0) = (1-sigma)δ(r+a), P_1(r,0) = sigma*δ(r+a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eig # Added for eigenvalue computation

# Parameters (Reverted to original request values for consistency)
k_0 = 1.0
Delta_mu = 10.0
f = 0.5
a = 1.0

# Sigma determines the initial chemical state. 
sigma = 1.0 

# Derived parameters
exp_plus = np.exp(Delta_mu / 2)
exp_minus = np.exp(-Delta_mu / 2)

# Spatial discretization - use a finer grid and wider domain
r_min = -5.0
r_max = 5.0
Nr = 1001  # Number of spatial points (odd to include r=0)
r = np.linspace(r_min, r_max, Nr)
dr = r[1] - r[0]

# Find indices closest to key positions
idx_plus_a = np.argmin(np.abs(r - a))
idx_minus_a = np.argmin(np.abs(r + a))

print(f"r[{idx_plus_a}] = {r[idx_plus_a]:.6f} (target: {a})")
print(f"r[{idx_minus_a}] = {r[idx_minus_a]:.6f} (target: {-a})")

# Time discretization
t_span = (0.0, 400.0)
t_eval = np.linspace(0.0, 400.0, 20000)

# Initial conditions: P_0(r,0) = (1-sigma)δ(r+a), P_1(r,0) = sigma*δ(r+a)
# Approximate delta function as a narrow Gaussian
sigma_delta = 1.0 * dr  # Width of delta approximation
P_delta = np.exp(-(r + a)**2 / (2 * sigma_delta**2)) / (sigma_delta * np.sqrt(2 * np.pi))
P_delta = P_delta / (np.sum(P_delta) * dr)  # Normalize to integrate to 1

P0_init = (1.0 - sigma) * P_delta
P1_init = sigma * P_delta

# Combine into single state vector
y0 = np.concatenate([P0_init, P1_init])

def compute_flux_divergence(r_arr, P, drift, dr):
    """
    Compute ∂_r ((r + drift)P + ∂_r P) using finite differences.
    This is the Fokker-Planck operator with position-dependent drift.
    """
    Nr = len(P)
    div_flux = np.zeros(Nr)
    
    # Interior points: use centered differences
    for i in range(1, Nr-1):
        # Compute flux at i+1/2
        dP_plus = (P[i+1] - P[i]) / dr
        flux_plus = (r_arr[i] + 0.5*dr + drift) * 0.5 * (P[i+1] + P[i]) + dP_plus
        
        # Compute flux at i-1/2
        dP_minus = (P[i] - P[i-1]) / dr
        flux_minus = (r_arr[i] - 0.5*dr + drift) * 0.5 * (P[i] + P[i-1]) + dP_minus
        
        # Divergence
        div_flux[i] = (flux_plus - flux_minus) / dr
    
    # Boundary conditions: zero flux at boundaries
    # Left boundary (i=0)
    dP_plus = (P[1] - P[0]) / dr
    flux_plus = (r_arr[0] + 0.5*dr + drift) * 0.5 * (P[1] + P[0]) + dP_plus
    div_flux[0] = flux_plus / dr
    
    # Right boundary (i=Nr-1)
    dP_minus = (P[-1] - P[-2]) / dr
    flux_minus = (r_arr[-1] - 0.5*dr + drift) * 0.5 * (P[-1] + P[-2]) + dP_minus
    div_flux[-1] = -flux_minus / dr
    
    return div_flux

def rhs(t, y):
    """
    Right-hand side of the coupled PDE system.
    y = [P0, P1] concatenated
    """
    Nr = len(y) // 2
    P0 = y[:Nr]
    P1 = y[Nr:]
    
    # Compute flux divergences
    dP0_dt = compute_flux_divergence(r, P0, f, dr)
    dP1_dt = compute_flux_divergence(r, P1, -f, dr)
    
    # Add delta function source terms at r = a (idx_plus_a)
    # For P_0: -k_0*e^{Δμ/2}δ(r-a)P_0 + k_0*e^{-Δμ/2}δ(r-a)P_1
    delta_contribution_0_at_a = (-k_0 * exp_plus * P0[idx_plus_a] + k_0 * exp_minus * P1[idx_plus_a]) / dr
    dP0_dt[idx_plus_a] += delta_contribution_0_at_a
    
    # For P_1: -k_0*e^{-Δμ/2}δ(r-a)P_1 + k_0*e^{Δμ/2}δ(r-a)P_0
    delta_contribution_1_at_a = (-k_0 * exp_minus * P1[idx_plus_a] + k_0 * exp_plus * P0[idx_plus_a]) / dr
    dP1_dt[idx_plus_a] += delta_contribution_1_at_a
    
    # Add delta function source terms at r = -a (idx_minus_a)
    # For P_0: -k_0*e^{-Δμ/2}δ(r+a)P_0
    delta_contribution_0_at_minus_a = (-k_0 * exp_minus * P0[idx_minus_a]) / dr
    dP0_dt[idx_minus_a] += delta_contribution_0_at_minus_a
    
    # For P_1: -k_0*e^{Δμ/2}δ(r+a)P_1
    delta_contribution_1_at_minus_a = (-k_0 * exp_plus * P1[idx_minus_a]) / dr
    dP1_dt[idx_minus_a] += delta_contribution_1_at_minus_a
    
    return np.concatenate([dP0_dt, dP1_dt])

# Solve the system
print("\nSolving coupled PDE system...")
print(f"Parameters: k_0={k_0}, Δμ={Delta_mu}, f={f}, a={a}, sigma={sigma}")
print(f"Spatial grid: [{r_min}, {r_max}] with {Nr} points (dr={dr:.6f})")
print(f"Time span: {t_span}, evaluating at {len(t_eval)} points")

sol = solve_ivp(rhs, t_span, y0, method='BDF', t_eval=t_eval, 
                rtol=1e-7, atol=1e-9, max_step=0.05)

print(f"Solution completed. Status: {sol.message}")

# Extract solutions
P0_sol = sol.y[:Nr, :]
P1_sol = sol.y[Nr:, :]

# Extract P_0(-a, t) and P_1(-a, t)
P0_at_minus_a = P0_sol[idx_minus_a, :]
P1_at_minus_a = P1_sol[idx_minus_a, :]

# Compute the functions to plot
f1_t = k_0 * exp_plus * P1_at_minus_a
f2_t = k_0 * exp_minus * P0_at_minus_a

# Save data to .dat file
data_to_save = np.vstack([sol.t, f1_t, f2_t]).T
header = 'Time\tf1(t)\tf2(t)'
np.savetxt('/path_to_folder/pde_solution_data.dat', data_to_save, fmt='%.6e', header=header, delimiter='\t')
print("\nData saved to 'pde_solution_data.dat'")

# Print some diagnostics
print(f"\nDiagnostics:")
print(f"P_0(-a, 0) = {P0_at_minus_a[0]:.6e}")
print(f"P_1(-a, 0) = {P1_at_minus_a[0]:.6e}")
print(f"P_0(-a, 10) = {P0_at_minus_a[-1]:.6e}")
print(f"P_1(-a, 10) = {P1_at_minus_a[-1]:.6e}")
print(f"f_1(0) = {f1_t[0]:.6e}")
print(f"f_2(0) = {f2_t[0]:.6e}")
print(f"f_1(10) = {f1_t[-1]:.6e}")
print(f"f_2(10) = {f2_t[-1]:.6e}")

# Check for conservation (total probability)
total_prob = np.trapz(P0_sol + P1_sol, r, axis=0)
print(f"\nTotal probability at t=0: {total_prob[0]:.6f}")
print(f"Total probability at t=10: {total_prob[-1]:.6f}")

# --- Eigenvalue Computation ---
print("\nComputing leading eigenvalue...")

def construct_operator_matrix(Nr, dr, r, f, k_0, exp_plus, exp_minus, idx_plus_a, idx_minus_a):
    """Construct the linear operator matrix M such that dY/dt = M * Y."""
    N = 2 * Nr
    M = np.zeros((N, N))
    
    # Create a wrapper for rhs that only takes y, as M is time-independent
    def rhs_y(y):
        return rhs(0, y)

    # Populate M column by column using basis vectors
    for j in range(N):
        e_j = np.zeros(N)
        e_j[j] = 1.0
        M[:, j] = rhs_y(e_j)
        
    return M

# Note: This is computationally expensive for large Nr.
# Nr=1001 means a 2002x2002 matrix.
M = construct_operator_matrix(Nr, dr, r, f, k_0, exp_plus, exp_minus, idx_plus_a, idx_minus_a)

# Compute eigenvalues
# Use only real part for comparison with decay rate
eigenvalues = eig(M, left=False, right=False) # Use scipy.linalg.eig for better performance
real_eigenvalues = np.real(eigenvalues)

# The leading eigenvalue is the one closest to zero (smallest magnitude, non-zero)
# Filter out the zero eigenvalue (which corresponds to the steady state if it exists)
# and find the smallest negative one.
negative_eigenvalues = real_eigenvalues[real_eigenvalues < -1e-9]
if len(negative_eigenvalues) > 0:
    lambda_0 = np.max(negative_eigenvalues)
    print(f"Leading Eigenvalue (lambda_0): {lambda_0:.6e}")
else:
    lambda_0 = None
    print("Could not find a negative leading eigenvalue.")


