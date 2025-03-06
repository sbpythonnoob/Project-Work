

import numpy as np
import matplotlib.pyplot as plt

def lorenz_deriv(t, state_perturb, sigma, rho, beta):
    x, y, z, dx, dy, dz = state_perturb
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    J = np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])
    ddxdt, ddydt, ddzdt = J @ np.array([dx, dy, dz])
    return np.array([dxdt, dydt, dzdt, ddxdt, ddydt, ddzdt])

def rk4_step(func, state_perturb, t, dt, *args):
    k1 = func(t, state_perturb, *args) * dt
    k2 = func(t + 0.5*dt, state_perturb + 0.5*k1, *args) * dt
    k3 = func(t + 0.5*dt, state_perturb + 0.5*k2, *args) * dt
    k4 = func(t + dt, state_perturb + k3, *args) * dt
    return state_perturb + (k1 + 2*k2 + 2*k3 + k4)/6

# Parameters for chaotic behavior
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Generate dummy BRICS market data using Lorenz system
def generate_dummy_data(steps=10000, dt=0.01):
    # Initial conditions (hypothetical market indices)
    initial_state = np.array([1.0, 1.0, 1.0])
    
    # Storage for dummy data (3 dimensions representing market dynamics)
    dummy_data = np.zeros((steps, 3))
    
    current_state = initial_state.copy()
    for i in range(steps):
        # Simple Euler integration for data generation
        x, y, z = current_state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        current_state += np.array([dx, dy, dz]) * dt
        dummy_data[i] = current_state
    
    return dummy_data

# Generate and visualize dummy data
dummy_brics_data = generate_dummy_data()
time = np.arange(dummy_brics_data.shape[0]) * 0.01

plt.figure(figsize=(12, 6))
plt.plot(time, dummy_brics_data[:, 0], label='Market Dimension X', alpha=0.7)
plt.plot(time, dummy_brics_data[:, 1], label='Market Dimension Y', alpha=0.7)
plt.plot(time, dummy_brics_data[:, 1], label='Market Dimension Z', alpha=0.7)
plt.title('BRICS Market Dummy Data (Lorenz System Simulation)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Market State')
plt.legend()
plt.show()

# Lyapunov exponent calculation with perturbation tracking
initial_perturbation = np.array([1.0, 0.0, 0.0])
initial_perturbation /= np.linalg.norm(initial_perturbation)

# Initialize combined state
state_perturb = np.concatenate((dummy_brics_data[0], initial_perturbation))

sum_logs = 0.0
lyapunov_history = []
dt = 0.01
steps = 10000

for i in range(1, steps):
    state_perturb = rk4_step(lorenz_deriv, state_perturb, 0, dt, sigma, rho, beta)
    
    # Extract perturbation vector
    dx, dy, dz = state_perturb[3:]
    current_norm = np.linalg.norm([dx, dy, dz])
    
    sum_logs += np.log(current_norm)
    state_perturb[3:] /= current_norm
    
    # Track running average of Lyapunov exponent
    if i % 100 == 0:
        lyapunov_history.append(sum_logs / (i * dt))

# Final calculation
lyapunov_exponent = sum_logs / (steps * dt)

# Plot convergence of Lyapunov estimate
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(lyapunov_history)) * 100 * dt, lyapunov_history)
plt.axhline(y=lyapunov_exponent, color='r', linestyle='--', label='Final Value')
plt.title('Lyapunov Exponent Convergence')
plt.xlabel('Time')
plt.ylabel('Estimated Exponent')
plt.legend()
plt.show()

print(f"\nFinal Largest Lyapunov Exponent: {lyapunov_exponent:.4f}")
print("Positive value indicates chaotic system characteristics")