# ==============================================================================
# Autonomous Ocean Microplastic Cleanup AUV Path Planner
# Student Name: Ranveer Ghadge | MIS Number: 612402067
# AI Algorithm: A* Search Algorithm (Cartesian Pathfinding)
# ML Model: K-Means Clustering (Acoustic Debris Sensor Analysis)
# ==============================================================================

# Import NumPy for numerical array operations and random seed generation
import numpy as np

# Import Matplotlib for rendering Cartesian 2D marine spatial visualizations
import matplotlib.pyplot as plt

# Import KMeans from scikit-learn for unsupervised machine learning clustering
from sklearn.cluster import KMeans

# Import heapq for priority queue (min-heap) graph exploration in A*
import heapq

# Set seed for reproducible marine grid generation
np.random.seed(101)

# ==============================================================================
# PART 1: A* SEARCH ALGORITHM & NODE CLASS DEFINITION
# ==============================================================================

class Node:
    """Represents a discrete state coordinate in the 2D Cartesian ocean grid."""
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position       # Cartesian coordinate tuple (x, y)
        self.g_cost = g_cost           # Distance navigated from start dock
        self.h_cost = h_cost           # Heuristic distance to target collection waypoint
        self.f_cost = g_cost + h_cost   # Total evaluation cost f(n) = g(n) + h(n)
        self.parent = parent           # Pointer to parent node for path reconstruction

    def __lt__(self, other):
        # Min-heap priority comparison operator based on lowest f_cost
        return self.f_cost < other.f_cost

def euclidean_heuristic(pos1, pos2):
    """Calculates Euclidean straight-line distance heuristic between Cartesian points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def a_star_search(grid, start, goal):
    """Core A* Algorithm finding optimal obstacle-free subsea trajectory."""
    rows, cols = grid.shape
    open_list = []
    closed_set = set()

    start_node = Node(start, 0.0, euclidean_heuristic(start, goal))
    heapq.heappush(open_list, start_node)
    g_costs = {start: 0.0}

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == goal:
            path = []
            curr = current_node
            while curr:
                path.append(curr.position)
                curr = curr.parent
            return path[::-1]

        closed_set.add(current_node.position)
        moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]

        for dx, dy in moves:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

            # 1. Grid boundary check
            if not (0 <= neighbor_pos[0] < rows and 0 <= neighbor_pos[1] < cols):
                continue
            # 2. Obstacle check (value == 1 means rocky ridge / trench)
            if grid[neighbor_pos[0], neighbor_pos[1]] == 1:
                continue
            # 3. Visited check
            if neighbor_pos in closed_set:
                continue

            step_cost = np.sqrt(dx**2 + dy**2)
            tentative_g = current_node.g_cost + step_cost

            if neighbor_pos not in g_costs or tentative_g < g_costs[neighbor_pos]:
                g_costs[neighbor_pos] = tentative_g
                h = euclidean_heuristic(neighbor_pos, goal)
                neighbor_node = Node(neighbor_pos, tentative_g, h, current_node)
                heapq.heappush(open_list, neighbor_node)

    return None

# ==============================================================================
# PART 2: MAIN SIMULATION & MACHINE LEARNING PIPELINE
# ==============================================================================

def main():
    print("=== Ocean Microplastic Cleanup AUV Simulation ===")
    print("Student: Ranveer Ghadge | MIS: 612402067")
    
    GRID_SIZE = 25
    NUM_HOTSPOTS = 55

    # Generate full Cartesian grid coordinates
    all_grid_coords = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1)]

    # Randomly select microplastic debris hotspots
    hotspot_indices = np.random.choice(len(all_grid_coords), NUM_HOTSPOTS, replace=False)
    debris_hotspots = np.array([all_grid_coords[i] for i in hotspot_indices])
    clear_zones = np.array([p for i, p in enumerate(all_grid_coords) if i not in set(hotspot_indices)])

    # Unsupervised ML Phase: K-Means Clustering
    NUM_CLUSTERS = 5
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=101, n_init=10)
    kmeans.fit(debris_hotspots)
    centroids = kmeans.cluster_centers_

    # Convert floating centroids to integer grid waypoints and sort left-to-right
    waypoints = [(int(round(c[0])), int(round(c[1]))) for c in centroids]
    waypoints.sort(key=lambda pt: pt[0])
    print(f"Target Collection Waypoints from K-Means: {waypoints}")

    # Subsea Grid & Obstacles
    auv_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    auv_grid[5:9, 10:13] = 1    # Rock Ridge / Trench 1
    auv_grid[14:18, 5:8] = 1    # Rock Outcrop 2

    # Classical AI Phase: Multi-Segment A* Search
    auv_base = (0, 0)
    current_pos = auv_base
    complete_auv_path = [current_pos]

    for wp in waypoints:
        sub_route = a_star_search(auv_grid, current_pos, wp)
        if sub_route:
            complete_auv_path.extend(sub_route[1:])
            current_pos = wp

    print(f"Path planning complete! Total subsea steps: {len(complete_auv_path)}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(plt.Rectangle((10, 5), 3, 4, color='navy', alpha=0.85, label="Subsea Obstacle (Rock Ridge/Trench)"))
    ax.add_patch(plt.Rectangle((5, 14), 3, 4, color='navy', alpha=0.85))

    ax.scatter(clear_zones[:, 0], clear_zones[:, 1], color='powderblue', s=25, alpha=0.4, label="Clean Water Columns")
    ax.scatter(debris_hotspots[:, 0], debris_hotspots[:, 1], c=kmeans.labels_, cmap="plasma", s=75, alpha=0.9, label="Microplastic Debris (Acoustic Sensor)")
    ax.scatter([w[0] for w in waypoints], [w[1] for w in waypoints], color="red", marker="X", s=220, zorder=5, label="Collection Target (A* Waypoint)")

    fx = [p[0] for p in complete_auv_path]
    fy = [p[1] for p in complete_auv_path]
    ax.plot(fx, fy, color="darkorange", linewidth=3, linestyle="-", marker="o", markersize=5, label="AUV Cleanup Flight Route")
    ax.scatter(auv_base[0], auv_base[1], color="green", s=300, marker="P", zorder=6, label="AUV Docking Station (Start)")

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(0, GRID_SIZE, 2))
    ax.set_yticks(range(0, GRID_SIZE, 2))
    ax.set_title("Ocean Microplastic Cleanup AUV Path Planner\n(K-Means Cluster Targeting + A* Cartesian Trajectory)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cartesian Grid X (Subsea Columns / Meters)")
    ax.set_ylabel("Cartesian Grid Y (Subsea Rows / Meters)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
