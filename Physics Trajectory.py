import numpy as np
import cupy as cp
from numba import cuda
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve
import math

G = 9.81  # gravity (m/s²)
L = 1.5   # pole length (m)


@cuda.jit
def compute_trajectory_kernel(x_vals, y_vals, h0, tan_theta, g, v0_sq, cos_theta_sq):
    """Compute y values for trajectory points."""
    idx = cuda.grid(1)
    if idx < x_vals.shape[0]:
        x = x_vals[idx]
        y_vals[idx] = h0 + x * tan_theta - (g * x * x) / (2.0 * v0_sq * cos_theta_sq)


@cuda.jit
def evaluate_polynomial_kernel(x_vals, y_vals, a, b, c):
    """Evaluate parabola: y = ax² + bx + c."""
    idx = cuda.grid(1)
    if idx < x_vals.shape[0]:
        x = x_vals[idx]
        y_vals[idx] = a * x * x + b * x + c


@cuda.jit
def batch_trajectory_kernel(x_vals, y_vals, coeffs, num_points):
    """Compute multiple trajectories at once."""
    traj_idx, point_idx = cuda.grid(2)
    if traj_idx < coeffs.shape[0] and point_idx < num_points:
        x = x_vals[point_idx]
        a = coeffs[traj_idx, 0]
        b = coeffs[traj_idx, 1]
        c = coeffs[traj_idx, 2]
        y_vals[traj_idx, point_idx] = a * x * x + b * x + c


@cuda.jit
def find_landing_kernel(y_vals, landing_indices):
    """Find where each trajectory hits the ground (y < 0)."""
    traj_idx = cuda.grid(1)
    if traj_idx < y_vals.shape[0]:
        landing_indices[traj_idx] = y_vals.shape[1] - 1
        for i in range(y_vals.shape[1]):
            if y_vals[traj_idx, i] < 0:
                landing_indices[traj_idx] = i
                break


class CUDATrajectorySimulator:
    """GPU-accelerated trajectory simulator."""
    
    def __init__(self, gravity=G, pole_length=L):
        self.g = gravity
        self.L = pole_length
        self.a_spline = None
        self.b_spline = None
        self.c_spline = None
        self.angles = []
        
        device = cuda.get_current_device()
        print(f"CUDA Device: {device.name.decode()}")
        print(f"Compute Capability: {device.compute_capability}")
        
    def compute_v0(self, theta_deg, range_):
        """Get initial velocity from angle and range."""
        theta = math.radians(theta_deg)
        h0 = self.L * math.sin(theta)
        
        def eqn(v0):
            t = (v0 * math.sin(theta) + np.sqrt((v0 * math.sin(theta))**2 + 2 * self.g * h0)) / self.g
            return v0 * math.cos(theta) * t - range_
        
        return fsolve(eqn, 10)[0]
    
    def fit_trajectories_gpu(self, trial_data, points_per_trajectory=100):
        """Fit parabola coefficients from trial data."""
        trial_data = sorted(trial_data)
        a_list, b_list, c_list = [], [], []
        self.angles = []
        
        all_x = []
        all_y = []
        
        for theta_deg, R in trial_data:
            theta = math.radians(theta_deg)
            h0 = self.L * math.sin(theta)
            v0 = self.compute_v0(theta_deg, R)
            
            x_vals_gpu = cp.linspace(0, R, points_per_trajectory, dtype=cp.float64)
            y_vals_gpu = cp.zeros(points_per_trajectory, dtype=cp.float64)
            
            threads_per_block = 256
            blocks = (points_per_trajectory + threads_per_block - 1) // threads_per_block
            
            compute_trajectory_kernel[blocks, threads_per_block](
                x_vals_gpu, y_vals_gpu,
                h0, math.tan(theta), self.g,
                v0**2, math.cos(theta)**2
            )
            
            cuda.synchronize()
            
            x_vals_cpu = cp.asnumpy(x_vals_gpu)
            y_vals_cpu = cp.asnumpy(y_vals_gpu)
            
            coeffs = np.polyfit(x_vals_cpu, y_vals_cpu, 2)
            a_list.append(coeffs[0])
            b_list.append(coeffs[1])
            c_list.append(coeffs[2])
            self.angles.append(theta_deg)
            
            all_x.append(x_vals_cpu)
            all_y.append(y_vals_cpu)
        
        self.a_spline = CubicSpline(self.angles, a_list, bc_type='natural')
        self.b_spline = CubicSpline(self.angles, b_list, bc_type='natural')
        self.c_spline = CubicSpline(self.angles, c_list, bc_type='natural')
        
        return all_x, all_y
    
    def predict_trajectory_gpu(self, theta_deg, x_max=1400, num_points=300):
        """Predict trajectory for one angle."""
        a = float(self.a_spline(theta_deg))
        b = float(self.b_spline(theta_deg))
        c = float(self.c_spline(theta_deg))
        
        x_vals_gpu = cp.linspace(0, x_max, num_points, dtype=cp.float64)
        y_vals_gpu = cp.zeros(num_points, dtype=cp.float64)
        
        threads_per_block = 256
        blocks = (num_points + threads_per_block - 1) // threads_per_block
        
        evaluate_polynomial_kernel[blocks, threads_per_block](
            x_vals_gpu, y_vals_gpu, a, b, c
        )
        
        cuda.synchronize()
        
        return cp.asnumpy(x_vals_gpu), cp.asnumpy(y_vals_gpu)
    
    def predict_batch_trajectories_gpu(self, angles_list, x_max=1400, num_points=300):
        """Predict trajectories for multiple angles at once."""
        num_trajectories = len(angles_list)
        
        coeffs = np.zeros((num_trajectories, 3), dtype=np.float64)
        for i, theta_deg in enumerate(angles_list):
            coeffs[i, 0] = float(self.a_spline(theta_deg))
            coeffs[i, 1] = float(self.b_spline(theta_deg))
            coeffs[i, 2] = float(self.c_spline(theta_deg))
        
        x_vals_gpu = cp.linspace(0, x_max, num_points, dtype=cp.float64)
        y_vals_gpu = cp.zeros((num_trajectories, num_points), dtype=cp.float64)
        coeffs_gpu = cp.asarray(coeffs)
        
        threads_per_block = (16, 16)
        blocks_x = (num_trajectories + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (num_points + threads_per_block[1] - 1) // threads_per_block[1]
        blocks = (blocks_x, blocks_y)
        
        batch_trajectory_kernel[blocks, threads_per_block](
            x_vals_gpu, y_vals_gpu, coeffs_gpu, num_points
        )
        
        cuda.synchronize()
        
        return cp.asnumpy(x_vals_gpu), cp.asnumpy(y_vals_gpu)
    
    def find_landing_ranges_gpu(self, x_vals, y_vals_batch):
        """Find where trajectories hit the ground."""
        num_trajectories = y_vals_batch.shape[0]
        
        y_vals_gpu = cp.asarray(y_vals_batch)
        landing_indices_gpu = cp.zeros(num_trajectories, dtype=cp.int32)
        
        threads_per_block = 256
        blocks = (num_trajectories + threads_per_block - 1) // threads_per_block
        
        find_landing_kernel[blocks, threads_per_block](
            y_vals_gpu, landing_indices_gpu
        )
        
        cuda.synchronize()
        
        landing_indices = cp.asnumpy(landing_indices_gpu)
        ranges = x_vals[landing_indices]
        
        return ranges, landing_indices


def main():
    print("=" * 60)
    print("CUDA-Accelerated Physics Trajectory Simulator")
    print("=" * 60)
    
    simulator = CUDATrajectorySimulator()
    
    print("\nEnter horizontal launch distances (in meters):")
    d15 = float(input("15°: "))
    d30 = float(input("30°: "))
    d45 = float(input("45°: "))
    d53 = float(input("53°: "))
    d27 = float(input("27°: "))
    
    trial_data = [(15, d15), (30, d30), (45, d45), (53, d53), (27, d27)]
    
    print("\nFitting trajectories on GPU...")
    simulator.fit_trajectories_gpu(trial_data)
    print("Trajectory fitting complete!")
    
    test_angle = float(input("\nEnter a launch angle to simulate: "))
    
    print(f"\nComputing trajectory for {test_angle}° on GPU...")
    x_vals, y_vals = simulator.predict_trajectory_gpu(test_angle)
    
    if np.any(y_vals < 0):
        landing_index = np.argmax(y_vals < 0)
        x_range = x_vals[landing_index]
    else:
        x_range = x_vals[-1]
    
    print(f"Estimated horizontal range at {test_angle}°: {x_range:.2f} meters")
    
    print("\nDemonstrating batch processing: Computing 36 trajectories in parallel...")
    batch_angles = np.linspace(15, 53, 36)
    x_batch, y_batch = simulator.predict_batch_trajectories_gpu(batch_angles, num_points=500)
    ranges, _ = simulator.find_landing_ranges_gpu(x_batch, y_batch)
    
    print(f"Computed {len(batch_angles)} trajectories in parallel!")
    print(f"Range statistics: min={ranges.min():.2f}m, max={ranges.max():.2f}m, mean={ranges.mean():.2f}m")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(x_vals, y_vals, 'b-', linewidth=2, label=f"{test_angle}° trajectory")
    axes[0].axvline(x_range, color='red', linestyle='--', label=f"Range ≈ {x_range:.2f} m")
    axes[0].set_title(f"Single Trajectory at {test_angle}° (GPU)")
    axes[0].set_xlabel("Horizontal Distance (m)")
    axes[0].set_ylabel("Vertical Height (m)")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    cmap = plt.cm.viridis
    for i, angle in enumerate(batch_angles):
        color = cmap(i / len(batch_angles))
        y_plot = np.clip(y_batch[i], 0, None)
        axes[1].plot(x_batch, y_plot, color=color, alpha=0.7, linewidth=1)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(batch_angles[0], batch_angles[-1]))
    plt.colorbar(sm, ax=axes[1], label='Launch Angle (°)')
    axes[1].set_title(f"Batch: {len(batch_angles)} Trajectories Computed in Parallel (GPU)")
    axes[1].set_xlabel("Horizontal Distance (m)")
    axes[1].set_ylabel("Vertical Height (m)")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
