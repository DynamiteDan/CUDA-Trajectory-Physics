import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
from math import sin, cos, tan, radians

# Constants
g = 9.81     # gravity (m/s²)
L = 1.5      # launch pole length (m)

# === Step 1: Input launch distances ===
print("Enter horizontal launch distances (in meters):")
d15 = float(input("15°: "))
d30 = float(input("30°: "))
d45 = float(input("45°: "))
d53 = float(input("53°: "))
d27 = float(input("27°: "))
trial_data = sorted([(15, d15), (30, d30), (45, d45), (53, d53), (27, d27)])

# === Step 2: Simulate true trajectory and fit parabola ===
def compute_v0(theta_deg, range_):
    theta = radians(theta_deg)
    h0 = L * sin(theta)
    def eqn(v0):
        t = (v0 * sin(theta) + np.sqrt((v0 * sin(theta))**2 + 2 * g * h0)) / g
        return v0 * cos(theta) * t - range_
    return fsolve(eqn, 10)[0]

angles = []
a_list, b_list, c_list = [], [], []

for theta_deg, R in trial_data:
    theta = radians(theta_deg)
    h0 = L * sin(theta)
    v0 = compute_v0(theta_deg, R)

    x_vals = np.linspace(0, R, 100)
    y_vals = h0 + x_vals * tan(theta) - (g * x_vals**2) / (2 * v0**2 * cos(theta)**2)
    
    coeffs = np.polyfit(x_vals, y_vals, 2)  # [a, b, c]
    a_list.append(coeffs[0])
    b_list.append(coeffs[1])
    c_list.append(coeffs[2])
    angles.append(theta_deg)

# === Step 3: Interpolate coefficients ===
a_spline = CubicSpline(angles, a_list, bc_type='natural')
b_spline = CubicSpline(angles, b_list, bc_type='natural')
c_spline = CubicSpline(angles, c_list, bc_type='natural')

# === Step 4: Predict trajectory ===
def predict_trajectory(theta_deg, x_max=1400, num_points=300):
    a = a_spline(theta_deg)
    b = b_spline(theta_deg)
    c = c_spline(theta_deg)
    x = np.linspace(0, x_max, num_points)
    y = a * x**2 + b * x + c
    return x, y

# === Step 5: Get test angle and compute ===
test_angle = float(input("\nEnter a launch angle to simulate: "))
x_vals, y_vals = predict_trajectory(test_angle)

# === Step 6: Estimate landing range ===
if np.any(y_vals < 0):
    landing_index = np.argmax(y_vals < 0)
    x_range = x_vals[landing_index]
else:
    x_range = x_vals[-1]

print(f"Estimated horizontal range at {test_angle}°: {x_range:.2f} meters")

# === Step 7: Plot ===
plt.plot(x_vals, y_vals, label=f"{test_angle}° trajectory")
plt.axvline(x_range, color='red', linestyle='--', label=f"Range ≈ {x_range:.2f} m")
plt.title(f"Trajectory Estimated from Launch Trials")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Height (m)")
plt.ylim(bottom=0, top=x_vals[-1])  # match y max to x max
plt.grid(True)
plt.legend()
plt.show()
