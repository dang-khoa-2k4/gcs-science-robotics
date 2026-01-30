import numpy as np
import pydot
import time
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pydrake.geometry.optimization import (
    Point,
)
from pydrake.solvers import (
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,   
)

from gcs.base import BaseGCS

class MultipleShootingGCS(BaseGCS):
    def __init__(self, regions, edges=None, path_weights=None, full_dim_overlap=False):
        BaseGCS.__init__(self, regions)

        if path_weights is None:
            path_weights = np.ones(self.dimension)
        elif isinstance(path_weights, float) or isinstance(path_weights, int):
            path_weights = path_weights * np.ones(self.dimension)
        assert len(path_weights) == self.dimension

        self.edge_cost = L2NormCost(
            np.hstack((np.diag(-path_weights), np.diag(path_weights))),
            np.zeros(self.dimension))

        for i, r in enumerate(self.regions):
            self.gcs.AddVertex(r, name = self.names[i] if not self.names is None else '')

        if edges is None:
            if full_dim_overlap:
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                edges = self.findEdgesViaOverlaps()

        vertices = self.gcs.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})")

            edge_length = edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), v.x())))

            # Constrain point in v to be in u
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def addSourceTarget(self, source, target, edges=None):
        source_edges, target_edges = super().addSourceTarget(source, target, edges)

        for edge in source_edges:
            for jj in range(self.dimension):
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

        for edge in target_edges:
            edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(edge.xu(), edge.xv())))


    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # Extract trajectory waypoints and corresponding regions
        waypoints = np.empty((self.dimension, 0))
        path_regions = []  # Lưu các vùng lồi trên đường đi
        
        for edge in best_path:
            new_waypoint = best_result.GetSolution(edge.xv())
            waypoints = np.concatenate(
                [waypoints, np.expand_dims(new_waypoint, 1)], axis=1)
            # Lấy vùng lồi của vertex đích
            path_regions.append(edge.v().set())

        traj = self.findMultipleShootingTraj(waypoints, path_regions, verbose)
        return traj, results_dict

    def findMultipleShootingTraj(self, waypoints, path_regions, verbose=False):
        """
        Multiple Shooting với ràng buộc vùng lồi từ GCS.
        
        Args:
            waypoints: Các điểm giao (facets) giữa vùng lồi từ GCS [dim x num_waypoints]
            path_regions: Danh sách các vùng lồi (ConvexSet) trên đường đi
            verbose: In thông tin debug
            obstacles: Danh sách obstacles để visualize (optional)
        
        Returns:
            dict chứa trajectory tối ưu
        """
        dt = getattr(self, 'dt', 0.1)  # Time step
        nx, nu = 4, 2  # Unicycle: [px, py, v, θ], [a, ω]
        
        num_regions = len(path_regions)
        intervals_per_region = getattr(self, 'intervals_per_region', 15)
        N = num_regions * intervals_per_region  # Tổng số shooting intervals
        
        # Điểm đầu và cuối
        x_start = np.array([waypoints[0, 0], waypoints[1, 0], 0.0, 0.0])
        x_goal = np.array([waypoints[0, -1], waypoints[1, -1], 0.0, 0.0])

        # ========== Định nghĩa động lực học Unicycle ==========
        x = ca.MX.sym('x', nx)
        u = ca.MX.sym('u', nu)
        
        px, py, v, theta = x[0], x[1], x[2], x[3]
        a, omega = u[0], u[1]

        # Lấy tham số từ config
        w_a = self.weights.get('a', 1.0) if hasattr(self, 'weights') else 1.0
        w_omega = self.weights.get('omega', 1.0) if hasattr(self, 'weights') else 1.0
        a_max = self.control_limits.get('a_max', 3.0) if hasattr(self, 'control_limits') else 3.0
        omega_max = self.control_limits.get('omega_max', np.pi) if hasattr(self, 'control_limits') else np.pi
        v_max = self.control_limits.get('v_max', 5.0) if hasattr(self, 'control_limits') else 5.0

        x_dot = ca.vertcat(
            v * ca.cos(theta),   # ṗx = v·cos(θ)
            v * ca.sin(theta),   # ṗy = v·sin(θ)
            a,                   # v̇ = a
            omega                # θ̇ = ω
        )
        f = ca.Function('f', [x, u], [x_dot])
        
        # RK4 integrator
        def rk4_step(f, x, u, dt):
            k1 = f(x, u)
            k2 = f(x + dt/2 * k1, u)
            k3 = f(x + dt/2 * k2, u)
            k4 = f(x + dt * k3, u)
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # ========== Xây dựng NLP (Multiple Shooting) ==========
        opti = ca.Opti()
        
        X = opti.variable(nx, N+1)   # Trạng thái tại N+1 điểm
        U = opti.variable(nu, N)     # Điều khiển tại N đoạn

        # Hàm mục tiêu: cực tiểu năng lượng điều khiển
        cost = 0
        for k in range(N):
            cost += w_a * U[0, k]**2 + w_omega * U[1, k]**2
        opti.minimize(cost * dt)

        # ========== RÀNG BUỘC ==========
        
        # 1. Ràng buộc biên (Boundary constraints)
        opti.subject_to(X[:, 0] == x_start)
        opti.subject_to(X[:2, N] == x_goal[:2])  # Chỉ ràng buộc vị trí cuối

        # 2. Ràng buộc động lực học (Defect/Continuity constraints)
        for k in range(N):
            x_next = rk4_step(f, X[:, k], U[:, k], dt)
            opti.subject_to(X[:, k+1] == x_next)

        # 3. RÀNG BUỘC VÙNG LỒI (Convex Region Constraints)
        # Mỗi nhóm đoạn k ∈ {n_i, ..., n_{i+1}} phải nằm trong vùng Q_i
        # Ràng buộc: A_i * [px, py] <= b_i
        for region_idx, region in enumerate(path_regions):
            # Xác định các node thuộc vùng này
            k_start = region_idx * intervals_per_region
            k_end = (region_idx + 1) * intervals_per_region
            
            # Bỏ qua nếu region là Point (source/target) - đã có boundary constraints
            if isinstance(region, Point):
                continue
            
            # Lấy ma trận A, b của vùng lồi (HPolyhedron)
            A = np.array(region.A())[:, :2]  # Chỉ lấy 2 cột đầu (px, py)
            b = np.array(region.b())
            
            for k in range(k_start, min(k_end + 1, N + 1)):
                # Ràng buộc: A * [px, py]^T <= b
                pos_k = X[:2, k]  # Chỉ vị trí [px, py]
                opti.subject_to(A @ pos_k <= b)

        # 4. Giới hạn điều khiển và trạng thái
        opti.subject_to(opti.bounded(-a_max, U[0, :], a_max))
        opti.subject_to(opti.bounded(-omega_max, U[1, :], omega_max))
        opti.subject_to(opti.bounded(0, X[2, :], v_max))  # v >= 0

        # ========== WARM START từ GCS waypoints ==========
        # Nội suy waypoints để khởi tạo cho tất cả N+1 nodes
        for k in range(N + 1):
            # Tìm waypoint gần nhất và nội suy
            region_idx = min(k // intervals_per_region, num_regions - 1)
            local_k = k - region_idx * intervals_per_region
            alpha = local_k / intervals_per_region
            
            if region_idx < num_regions - 1:
                # Nội suy giữa waypoint[region_idx] và waypoint[region_idx+1]
                wp_start = waypoints[:, region_idx]
                wp_end = waypoints[:, region_idx + 1]
            else:
                # Đoạn cuối: nội suy đến goal
                wp_start = waypoints[:, -1] if region_idx < waypoints.shape[1] else waypoints[:, -1]
                wp_end = x_goal[:2]
            
            pos_init = (1 - alpha) * wp_start + alpha * wp_end
            x_init = np.array([pos_init[0], pos_init[1], 0.5, 0.0])
            opti.set_initial(X[:, k], x_init)
        
        opti.set_initial(U, 0)

        # ========== Lưu lại quá trình hội tụ ==========
        iteration_history = []
        
        def callback(i):
            """Callback để lưu trạng thái mỗi iteration"""
            try:
                X_current = opti.debug.value(X)
                cost_current = opti.debug.value(cost * dt)
                iteration_history.append({
                    'iter': len(iteration_history),
                    'X': X_current.copy(),
                    'cost': cost_current
                })
            except:
                pass
            return False  # Tiếp tục tối ưu
        
        opti.callback(callback)

        # ========== Giải NLP ==========
        opts = {'ipopt': {'print_level': 3 if verbose else 0, 'max_iter': 500}}
        opti.solver('ipopt', opts)

        result = {}
        try:
            sol = opti.solve()
            result['success'] = True
            result['X'] = sol.value(X)
            result['U'] = sol.value(U)
            result['cost'] = sol.value(cost * dt)
            result['dt'] = dt
            result['N'] = N
            if verbose:
                print(f"MS solved: cost={result['cost']:.4f}, N={N} intervals")
        except Exception as e:
            result['success'] = False
            result['X'] = opti.debug.value(X)
            result['U'] = opti.debug.value(U)
            result['error'] = str(e)
            if verbose:
                print(f"MS failed: {e}")

        result['iteration_history'] = iteration_history
    
        return result

        
    def addLimitControl(self, control_limits):
        """Set control limits: {'a_max': ..., 'omega_max': ..., 'v_max': ...}"""
        self.control_limits = control_limits

    def addWeightsCost(self, weights):
        """Set cost weights: {'a': ..., 'omega': ...}"""
        self.weights = weights

    def setMSParams(self, dt=0.1, intervals_per_region=5):
        """Set Multiple Shooting parameters.
        
        Args:
            dt: Time step for integration
            intervals_per_region: Number of shooting intervals per convex region
        """
        self.dt = dt
        self.intervals_per_region = intervals_per_region


    def visualize_ms_result(self, result, x_start, x_goal, obstacles=None, 
                            iteration_history=None, save_path=None, show=True):
        """
        Visualize Multiple Shooting trajectory và quá trình hội tụ.
        
        Args:
            result: dict chứa {'X': states, 'U': controls, 'dt': timestep, 'N': intervals}
            x_start: Điểm đầu [px, py, v, theta]
            x_goal: Điểm cuối [px, py, v, theta]
            obstacles: List các vật cản, mỗi vật cản là dict {toa do tao nen}
            iteration_history: List các dict {'iter': i, 'X': states, 'cost': cost} để vẽ hội tụ
            save_path: Đường dẫn lưu file (không có extension)
            show: Hiển thị plot hay không
        
        Returns:
            dict chứa các figure đã tạo
        """
        X_opt = result['X']
        U_opt = result['U']
        dt = result.get('dt', 0.1)
        N = result.get('N', U_opt.shape[1])
        
        figures = {}
        
        # ========== Plot 1: Trajectory và Controls ==========
        fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Trajectory
        ax = axes[0]
        if obstacles:
            for O in obstacles:
                ax.fill(*O.T, fc='lightcoral', ec='k', zorder=4)
        
        ax.plot(X_opt[0, :], X_opt[1, :], 'b.-', linewidth=2, label='Trajectory')
        ax.plot(x_start[0], x_start[1], 'go', markersize=10, label='Start')
        ax.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal')
        
        # Vẽ hướng robot
        for i in range(0, N+1, max(1, N//10)):
            px, py, v, theta = X_opt[:, i]
            dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
            ax.arrow(px, py, dx, dy, head_width=0.15, head_length=0.1,
                    fc='green', ec='green', alpha=0.7)
        
        ax.set_xlabel('$p_x$')
        ax.set_ylabel('$p_y$')
        ax.set_title('Multiple Shooting Trajectory')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # Controls
        ax = axes[1]
        t = np.arange(N) * dt
        ax.step(t, U_opt[0, :], 'b-', where='post', label='Acceleration $a$')
        ax.step(t, U_opt[1, :], 'r-', where='post', label='Angular velocity $\\omega$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Controls')
        ax.set_title('Control Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['trajectory'] = fig1
        
        if save_path:
            fig1.savefig(f'{save_path}_trajectory.png', dpi=150)
        
        # ========== Plot 2: Convergence (nếu có iteration_history) ==========
        if iteration_history and len(iteration_history) > 1:
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
            
            # Quỹ đạo qua các iteration
            ax1 = axes2[0]
            if obstacles:
                for O in obstacles:
                    ax1.fill(*O.T, fc='lightcoral', ec='k', zorder=4)
            
            ax1.plot(x_start[0], x_start[1], 'go', markersize=12, label='Start', zorder=5)
            ax1.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal', zorder=5)
            
            n_show = min(8, len(iteration_history))
            indices = np.linspace(0, len(iteration_history)-1, n_show, dtype=int)
            colors = plt.cm.viridis(np.linspace(0, 1, n_show))
            
            for i, idx in enumerate(indices):
                X_iter = iteration_history[idx]['X']
                alpha = 0.3 + 0.7 * (i / (n_show - 1)) if n_show > 1 else 1.0
                lw = 1 + 2 * (i / (n_show - 1)) if n_show > 1 else 2
                ax1.plot(X_iter[0, :], X_iter[1, :], 'o-', color=colors[i],
                        linewidth=lw, alpha=alpha, markersize=3, label=f'Iter {idx}')
            
            ax1.set_xlabel('$p_x$')
            ax1.set_ylabel('$p_y$')
            ax1.set_title('Trajectory Convergence')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.axis('equal')
            ax1.grid(True, alpha=0.3)
            
            # Cost convergence
            ax2 = axes2[1]
            iters = [h['iter'] for h in iteration_history]
            costs = [h['cost'] for h in iteration_history]
            ax2.semilogy(iters, costs, 'b.-', linewidth=2, markersize=4)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Cost (log scale)')
            ax2.set_title('Cost Convergence')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=costs[-1], color='r', linestyle='--', alpha=0.5,
                    label=f'Final: {costs[-1]:.4f}')
            ax2.legend()
            
            plt.tight_layout()
            figures['convergence'] = fig2
            
            if save_path:
                fig2.savefig(f'{save_path}_convergence.png', dpi=150)
            
            # ========== Animation ==========
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            
            def init():
                ax3.clear()
                return []
            
            def animate(frame):
                ax3.clear()
                if obstacles:
                    for O in obstacles:
                        ax3.fill(*O.T, fc='lightcoral', ec='k', zorder=4)
                
                ax3.plot(x_start[0], x_start[1], 'go', markersize=12)
                ax3.plot(x_goal[0], x_goal[1], 'r*', markersize=15)
                
                X_iter = iteration_history[frame]['X']
                ax3.plot(X_iter[0, :], X_iter[1, :], 'b.-', linewidth=2, markersize=5)
                
                # Vẽ hướng robot
                n_pts = X_iter.shape[1]
                for i in range(0, n_pts, max(1, n_pts//10)):
                    px, py, v, theta = X_iter[:, i]
                    dx, dy = 0.3 * np.cos(theta), 0.3 * np.sin(theta)
                    ax3.arrow(px, py, dx, dy, head_width=0.15, head_length=0.1,
                            fc='green', ec='green', alpha=0.7)
                
                # Auto scale
                all_x = X_iter[0, :]
                all_y = X_iter[1, :]
                margin = 2
                ax3.set_xlim(min(all_x.min(), x_start[0], x_goal[0]) - margin,
                            max(all_x.max(), x_start[0], x_goal[0]) + margin)
                ax3.set_ylim(min(all_y.min(), x_start[1], x_goal[1]) - margin,
                            max(all_y.max(), x_start[1], x_goal[1]) + margin)
                ax3.set_aspect('equal')
                ax3.grid(True, alpha=0.3)
                ax3.set_title(f'Iteration {frame} | Cost: {iteration_history[frame]["cost"]:.4f}')
                return []
            
            anim = FuncAnimation(fig3, animate, init_func=init,
                                frames=len(iteration_history), interval=200, blit=False)
            figures['animation'] = anim
            
            if save_path:
                anim.save(f'{save_path}_animation.gif', writer='pillow', fps=5)
            
            print(f"\nTotal iterations: {len(iteration_history)}")
            print(f"Cost: {costs[-1]:.4f}")
        
        if show:
            plt.show()
        
        return figures


    def visualize_ms_with_regions(self,result, x_start, x_goal, regions, 
                                obstacles=None, save_path=None, show=True):
        """
        Visualize trajectory cùng với các vùng lồi từ GCS.
        
        Args:
            result: dict chứa trajectory
            x_start, x_goal: Điểm đầu/cuối
            regions: List các HPolyhedron (vùng lồi)
            obstacles: List các vật cản
            save_path: Đường dẫn lưu
            show: Hiển thị hay không
        """
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        
        X_opt = result['X']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Vẽ vùng lồi
        patches = []
        for i, region in enumerate(regions):
            # Lấy vertices của polytope (cần sample các điểm)
            A, b = np.array(region.A()), np.array(region.b())
            # Vẽ đơn giản bằng cách sample
            from scipy.spatial import ConvexHull
            # Sample points trong region
            try:
                center = np.linalg.lstsq(A, b, rcond=None)[0][:2]
                angles = np.linspace(0, 2*np.pi, 50)
                pts = []
                for angle in angles:
                    d = np.array([np.cos(angle), np.sin(angle)])
                    # Tìm điểm xa nhất theo hướng d trong region
                    t_max = np.inf
                    for j in range(len(b)):
                        a_j = A[j, :2]
                        denom = a_j @ d
                        if denom > 1e-6:
                            t = (b[j] - a_j @ center) / denom
                            t_max = min(t_max, t)
                    if t_max < np.inf and t_max > 0:
                        pts.append(center + t_max * d)
                if len(pts) >= 3:
                    pts = np.array(pts)
                    hull = ConvexHull(pts)
                    poly = Polygon(pts[hull.vertices], alpha=0.2, 
                                facecolor=plt.cm.Set3(i % 12), edgecolor='black')
                    ax.add_patch(poly)
            except:
                pass
        
        # Vẽ vật cản
        if obstacles:
            for obs in obstacles:
                circle = plt.Circle((obs['x'], obs['y']), obs['r'],
                                color='red', alpha=0.5)
                ax.add_patch(circle)
        
        # Vẽ trajectory
        ax.plot(X_opt[0, :], X_opt[1, :], 'b.-', linewidth=2, label='Trajectory')
        ax.plot(x_start[0], x_start[1], 'go', markersize=12, label='Start')
        ax.plot(x_goal[0], x_goal[1], 'r*', markersize=15, label='Goal')
        
        ax.set_xlabel('$p_x$')
        ax.set_ylabel('$p_y$')
        ax.set_title('Multiple Shooting with Convex Regions')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f'{save_path}_regions.png', dpi=150)
        
        if show:
            plt.show()
        
        return fig