import numpy as np
import casadi as ca
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 45       # Number of shooting intervals
dt = 0.1            # Time step
nx, nu = 4, 2       # Number of state and control variables

# Start and goal points
x_start = np.array([0.0, 0.0, 0.0, 0.0])      # [px, py, v, theta]
x_goal = np.array([10.0, 10.0, 0.0, 0.0])

# obstacle 
obs_x, obs_y, R_obs = 5.0, 5.0, 2.0
R_robot = 0.3
R_safe = R_obs + R_robot

# limit controls
a_max, omega_max = 3.0, np.pi
v_max = 8.0

# weights for cost function
w_a, w_omega = 1.0, 1.0

x = ca.MX.sym('x', nx)
u = ca.MX.sym('u', nu)

# x_dot = f(x, u): Unicycle model
px, py, v, theta = x[0], x[1], x[2], x[3]
a, omega = u[0], u[1]

x_dot = ca.vertcat(
    v * ca.cos(theta),   # px_dot = v·cos(theta)
    v * ca.sin(theta),   # py_dot = v·sin(theta)
    a,                   # v_dot = a
    omega                # theta_dot = omega
)
f = ca.Function('f', [x, u], [x_dot])

# RK4 integration step
def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def rk4_intermediate_points(f, x, u, dt):
    """Trả về các điểm trung gian trong bước RK4 để kiểm tra va chạm."""
    k1 = f(x, u)
    x1 = x + dt/2 * k1          # Điểm tại t + dt/2 (lần 1)
    k2 = f(x1, u)
    x2 = x + dt/2 * k2          # Điểm tại t + dt/2 (lần 2)  
    k3 = f(x2, u)
    x3 = x + dt * k3            # Điểm tại t + dt (trước khi trung bình)
    return [x1, x2, x3]

opti = ca.Opti()

# decision variables: states at each node and controls at each interval
X = opti.variable(nx, N+1)   # States at N+1 points
U = opti.variable(nu, N)     # Controls at N intervals

# objective function: minimize control energy
cost = 0
for k in range(N):
    cost += w_a * U[0, k]**2 + w_omega * U[1, k]**2
opti.minimize(cost * dt)


# Boundary constraints
opti.subject_to(X[:, 0] == x_start)
opti.subject_to(X[:, N] == x_goal[:])  

# Dynamics constraints - Continuity constraint
for k in range(N):
    x_next = rk4_step(f, X[:, k], U[:, k], dt)
    opti.subject_to(X[:, k+1] == x_next)  

# Obstacle constraints
for k in range(N+1):
    dist_sq = (X[0, k] - obs_x)**2 + (X[1, k] - obs_y)**2
    opti.subject_to(dist_sq >= R_safe**2)

# Control and state limits
opti.subject_to(opti.bounded(-a_max, U[0, :], a_max))
opti.subject_to(opti.bounded(-omega_max, U[1, :], omega_max))
opti.subject_to(opti.bounded(-v_max, X[2, :], v_max))


# Straight line from A to B as initial guess
for k in range(N+1):
    alpha = k / N
    x_init = (1-alpha)*x_start + alpha*x_goal
    opti.set_initial(X[:, k], x_init)
opti.set_initial(U, 0)

# ========== Callback to store iteration history ==========
iteration_history = []

def callback(i):
    """Callback để lưu trạng thái mỗi iteration"""
    X_current = opti.debug.value(X)
    cost_current = opti.debug.value(cost * dt)
    iteration_history.append({
        'iter': len(iteration_history),
        'X': X_current.copy(),
        'cost': cost_current
    })
    return False  # Tiếp tục tối ưu

opti.callback(callback)

opti.solver('ipopt', {'ipopt': {'print_level': 3, 'max_iter': 500}})

try:
    sol = opti.solve()
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    print("Found")
except:
    print("Solver did not converge, using debug solution.")
    X_opt = opti.debug.value(X)
    U_opt = opti.debug.value(U)

# ========== Plot results ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Trajectory
ax = axes[0]
circle = plt.Circle((obs_x, obs_y), R_obs, color='red', alpha=0.5, label='Obstacle')
ax.add_patch(circle)
ax.plot(X_opt[0, :], X_opt[1, :], 'b.-', linewidth=2, label='Trajectory')
ax.plot(x_start[0], x_start[1], 'go', markersize=10, label='Start')
ax.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal')
ax.set_xlabel('$p_x$')
ax.set_ylabel('$p_y$')
ax.set_title('Multiple Shooting: Avoiding Obstacle')
ax.legend()
ax.axis('equal')
ax.grid(True)

# Controls
ax = axes[1]
t = np.arange(N) * dt
ax.step(t, U_opt[0, :], 'b-', where='post', label='Acceleration $a$')
ax.step(t, U_opt[1, :], 'r-', where='post', label='Angular velocity $omega$')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Controls')
ax.set_title('Control signals')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('multiple_shooting_result.png', dpi=150)
plt.show()

print(f"\nTotal cost: {sol.value(cost * dt):.4f}")

if len(iteration_history) > 1:
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Animation của quỹ đạo qua các iteration
    ax1 = axes2[0]
    circle = plt.Circle((obs_x, obs_y), R_obs, color='red', alpha=0.3)
    ax1.add_patch(circle)
    ax1.plot(x_start[0], x_start[1], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal', zorder=5)
    
    # Vẽ một số iteration quan trọng
    n_show = min(8, len(iteration_history))
    indices = np.linspace(0, len(iteration_history)-1, n_show, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))
    
    for i, idx in enumerate(indices):
        X_iter = iteration_history[idx]['X']
        alpha = 0.3 + 0.7 * (i / (n_show - 1)) if n_show > 1 else 1.0
        lw = 1 + 2 * (i / (n_show - 1)) if n_show > 1 else 2
        ax1.plot(X_iter[0, :], X_iter[1, :], 'o-', color=colors[i], 
                 linewidth=lw, alpha=alpha, markersize=3,
                 label=f'Iter {idx}')
    
    ax1.set_xlabel('$p_x$')
    ax1.set_ylabel('$p_y$')
    ax1.set_title('Quá trình hội tụ của quỹ đạo')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 12)
    
    # Plot 2: Cost function qua các iteration
    ax2 = axes2[1]
    iters = [h['iter'] for h in iteration_history]
    costs = [h['cost'] for h in iteration_history]
    ax2.semilogy(iters, costs, 'b.-', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost (log scale)')
    ax2.set_title('Hội tụ của hàm mục tiêu')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=costs[-1], color='r', linestyle='--', alpha=0.5, 
                label=f'Final: {costs[-1]:.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_visualization.png', dpi=150)
    plt.show()
    
    # ========== Animation ==========
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    
    def init():
        ax3.clear()
        return []
    
    def animate(frame):
        ax3.clear()
        circle = plt.Circle((obs_x, obs_y), R_obs, color='red', alpha=0.4)
        ax3.add_patch(circle)
        ax3.plot(x_start[0], x_start[1], 'go', markersize=12)
        ax3.plot(x_goal[0], x_goal[1], 'r*', markersize=15)
        
        X_iter = iteration_history[frame]['X']
        ax3.plot(X_iter[0, :], X_iter[1, :], 'b.-', linewidth=2, markersize=5)

        for i in range(0, N+1, 5):
            px, py, v, theta = X_iter[:, i]
            dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
            ax3.arrow(px, py, dx, dy, head_width=0.15, head_length=0.1, 
                     fc='green', ec='green', alpha=0.7)
        
        ax3.set_xlim(-1, 12)
        ax3.set_ylim(-1, 12)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_title(f'Iteration {frame} | Cost: {iteration_history[frame]["cost"]:.4f}')
        return []
    
    anim = FuncAnimation(fig3, animate, init_func=init, 
                        frames=len(iteration_history), interval=200, blit=False)
    anim.save('convergence_animation.gif', writer='pillow', fps=5)
    plt.show()
    
    print(f"\nTổng số iterations: {len(iteration_history)}")
    print(f"Cost: {costs[-1]:.4f}")
else:
    print("Không đủ iteration để visualize")
