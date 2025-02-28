import numpy as np
import matplotlib.pyplot as plt

# Define the amino acid letters (order consistent with one-hot encoding)
aa_letters = list('ACDEFGHIKLMNPQRSTVWY')

# -------------------------------
# Create a 9x20 Original One-Hot Matrix
# -------------------------------
# Each of the 9 rows corresponds to a sequence position.
# For demonstration, we assign a "hot" (1) at a chosen index per row.
original_matrix = np.zeros((9, 20))
one_hot_positions = [3, 7, 12, 0, 15, 5, 10, 18, 8]  # example positions for each sequence position
for i, pos in enumerate(one_hot_positions):
    original_matrix[i, pos] = 1

# -------------------------------
# Simulate a Diffused Matrix
# -------------------------------
# We'll mimic the heat diffusion process by applying simple smoothing.
diffused_matrix = original_matrix.copy().astype(float)

# Horizontal smoothing: average each element with its immediate horizontal neighbors.
temp_matrix = diffused_matrix.copy()
for i in range(9):
    for j in range(20):
        left = max(j - 1, 0)
        right = min(j + 2, 20)  # right is non-inclusive
        temp_matrix[i, j] = np.mean(diffused_matrix[i, left:right])

# Vertical smoothing: average each element with its immediate vertical neighbors.
for i in range(9):
    for j in range(20):
        top = max(i - 1, 0)
        bottom = min(i + 2, 9)
        diffused_matrix[i, j] = np.mean(temp_matrix[top:bottom, j])

# -------------------------------
# Plotting Side-by-Side Heat Maps
# -------------------------------
# We transpose the matrices so that:
#   - x-axis: sequence positions (9 columns)
#   - y-axis: amino acids (20 rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original matrix heat map
im0 = axes[0].imshow(original_matrix.T, cmap='viridis', aspect='auto')
axes[0].set_title("Original One-Hot Matrix (9x20)")
axes[0].set_xlabel("Sequence Position")
axes[0].set_ylabel("Amino Acid")
axes[0].set_xticks(range(9))
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(aa_letters)
fig.colorbar(im0, ax=axes[0], label="Intensity")

# Diffused matrix heat map
im1 = axes[1].imshow(diffused_matrix.T, cmap='viridis', aspect='auto')
axes[1].set_title("Diffused Matrix (9x20)")
axes[1].set_xlabel("Sequence Position")
axes[1].set_ylabel("Amino Acid")
axes[1].set_xticks(range(9))
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(aa_letters)
fig.colorbar(im1, ax=axes[1], label="Intensity")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def solve_wave_equation_2d(u, dt, dx, dy, c, steps):
    """
    Simplified 2D wave equation solver.
    """
    nx, ny = u.shape
    u_next = np.zeros_like(u)
    u_prev = np.zeros_like(u)

    for _ in range(steps):
        u_next[1:-1, 1:-1] = (2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] +
                              (c * dt / dx) ** 2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) +
                              (c * dt / dy) ** 2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]))
        u_prev, u = u, u_next

    return u

# Generate random 9x20 "before" wave matrix
np.random.seed(42)
X_before = np.random.rand(9, 20)

# Apply wave equation to generate "after" wave matrix
dt = 0.1
dx = 1.0
dy = 1.0
c = 5.0  # Wave speed
steps = 10  # Number of time steps
X_after = solve_wave_equation_2d(X_before, dt, dx, dy, c, steps)

# Plot the "before" and "after" matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before
cax1 = axes[0].imshow(X_before, cmap='viridis', aspect='auto')
axes[0].set_title("Before Wave Equation")
axes[0].set_xlabel("Amino Acid Position")
axes[0].set_ylabel("Sequence Position")
fig.colorbar(cax1, ax=axes[0])

# After
cax2 = axes[1].imshow(X_after, cmap='viridis', aspect='auto')
axes[1].set_title("After Wave Equation")
axes[1].set_xlabel("Amino Acid Position")
axes[1].set_ylabel("Sequence Position")
fig.colorbar(cax2, ax=axes[1])

plt.tight_layout()
plt.show()

# Calculate the difference between the before and after states
difference_matrix = X_after - X_before
print("Difference Matrix (After - Before):")
print(difference_matrix)

import numpy as np
import matplotlib.pyplot as plt

# Create a 1D temperature field with a peak
x = np.linspace(0, 10, 100)
u = np.exp(-(x - 5)**2)  # Gaussian peak at x = 5

# Compute the Laplacian (second derivative)
laplacian = np.gradient(np.gradient(u, x), x)

# Plot the temperature field and Laplacian
plt.figure(figsize=(10, 6))
plt.plot(x, u, label='Temperature Field (u)', color='blue', linewidth=2)
plt.plot(x, laplacian, label='Laplacian (∇²u)', color='red', linestyle='--', linewidth=2)
plt.title('Temperature Field and Laplacian', fontsize=14)
plt.xlabel('Position (x)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()