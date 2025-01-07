import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

# --- Create an 'S'-shaped manifold ---
def generate_s_shape(n_points=500, noise_std=0.01):
    """
    Generates an 'S' shape in 3D. 
    Adds slight Gaussian noise for realism.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    # Classic S: x = sin(t), y flips halfway, z ~ t
    # We reduce the amplitude from 0.5 to 0.2, so the fold isn't so large.
    x = np.sin(t)
    y = 0.2 * np.sign(t - np.pi) * (1 - np.cos(t))  
    # Shrink z-range from ~[0,2] to something smaller, e.g. ~[0,1.2]
    z = 0.05 * (t / np.pi)  
    
    # Add slight Gaussian noise
    noise = np.random.normal(0, noise_std, (n_points, 3))
    return np.column_stack([x, y, z]) + noise

# --- Create a linear manifold (plane) ---
def generate_linear_fold(n_points=484):
    """
    Generates a flat plane in 3D.
    By default, n_points=484 (22x22), so indices from 0 to 483 are valid.
    """
    side = int(np.round(np.sqrt(n_points)))
    # If n_points is not a perfect square, you can trim or reshape:
    # But for 484 = 22^2, this works perfectly.
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, side),
        np.linspace(0, 1, side)
    )
    grid_z = np.zeros_like(grid_x)
    plane_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    
    # If you really want exactly n_points even if it isn't square:
    # plane_points = plane_points[:n_points]
    return plane_points

# --- Shortest path helper: reconstruct path from predecessors ---
def reconstruct_path(predecessors, start_idx, goal_idx):
    """
    Reconstructs the path (as a list of indices) from 'goal_idx' back to 'start_idx'
    using the predecessor matrix. Returns an empty list if no path is found.
    """
    path = []
    current = goal_idx
    
    # -9999 indicates "no predecessor" in scipy’s shortest_path
    while (current != -9999) and (current != start_idx):
        path.append(current)
        current = int(predecessors[start_idx, current])
    
    # Check if we actually reached the start node
    if current == start_idx:
        path.append(start_idx)
        path.reverse()
        return path
    else:
        return []  # No valid path found

# --- Plot S-shape ---
def plot_s_shape():
    s_shape = generate_s_shape(n_points=500)

    # Use a reasonably large k to keep the graph connected
    k = 10
    graph = kneighbors_graph(s_shape, n_neighbors=k, mode='distance', include_self=False)
    # Compute all shortest paths
    manifold_distances, predecessors = shortest_path(graph, directed=False, return_predecessors=True)

    # Choose two points
    point_a_idx = 100
    point_b_idx = 300
    
    point_a = s_shape[point_a_idx]
    point_b = s_shape[point_b_idx]

    # Compute Euclidean (L2) distance
    l2_distance = np.linalg.norm(point_a - point_b)

    # Compute manifold (graph) distance
    manifold_distance = manifold_distances[point_a_idx, point_b_idx]

    # Check if there is a valid path
    if np.isinf(manifold_distance):
        print(f"No path found between {point_a_idx} and {point_b_idx}. Try increasing k.")
        path_coords = np.vstack([point_a, point_b])
    else:
        path_indices = reconstruct_path(predecessors, point_a_idx, point_b_idx)
        path_coords = s_shape[path_indices]

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter all points
    ax.scatter(s_shape[:, 0], s_shape[:, 1], s_shape[:, 2],
               label="S-shaped manifold", color="blue", alpha=0.6)

    # Highlight chosen points
    ax.scatter(*point_a, color='red', label="Point A")
    ax.scatter(*point_b, color='green', label="Point B")

    # Draw the straight (L2) line
    ax.plot([point_a[0], point_b[0]],
            [point_a[1], point_b[1]],
            [point_a[2], point_b[2]],
            linestyle='--', color='red',
            label=f"L2 = {l2_distance:.2f}")

    # Draw manifold path if valid
    if len(path_coords) > 1:
        ax.plot(path_coords[:, 0],
                path_coords[:, 1],
                path_coords[:, 2],
                color='purple',
                label=f"Manifold = {manifold_distance:.2f}")

    ax.text(*point_a, "A", color='red')
    ax.text(*point_b, "B", color='green')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("S-shaped Manifold Distance vs L2 Distance")

    plt.show()

# --- Plot linear fold ---
def plot_linear_fold():
    linear_fold = generate_linear_fold(n_points=484)

    # For a grid of 22x22, k=10 may or may not keep the whole graph connected.
    # Feel free to try k=15 or k=20 if you get no path.
    k = 10
    graph = kneighbors_graph(linear_fold, n_neighbors=k, mode='distance', include_self=False)
    # Compute all shortest paths
    manifold_distances, predecessors = shortest_path(graph, directed=False, return_predecessors=True)

    # Choose two points
    point_a_idx = 10
    point_b_idx = 300  # valid because we have 0..483

    point_a = linear_fold[point_a_idx]
    point_b = linear_fold[point_b_idx]

    # Euclidean distance
    l2_distance = np.linalg.norm(point_a - point_b)

    # Manifold distance
    manifold_distance = manifold_distances[point_a_idx, point_b_idx]

    if np.isinf(manifold_distance):
        print(f"No path found between {point_a_idx} and {point_b_idx}. Try increasing k.")
        path_coords = np.vstack([point_a, point_b])
    else:
        path_indices = reconstruct_path(predecessors, point_a_idx, point_b_idx)
        path_coords = linear_fold[path_indices]

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(linear_fold[:, 0], linear_fold[:, 1], linear_fold[:, 2],
               label="Linear manifold", color="orange", alpha=0.6)

    # Points and direct L2 line
    ax.scatter(*point_a, color='red', label="Point A")
    ax.scatter(*point_b, color='green', label="Point B")
    ax.plot([point_a[0], point_b[0]],
            [point_a[1], point_b[1]],
            [point_a[2], point_b[2]],
            linestyle='--', color='red',
            label=f"L2 = {l2_distance:.2f}")

    # Manifold path
    if len(path_coords) > 1:
        ax.plot(path_coords[:, 0],
                path_coords[:, 1],
                path_coords[:, 2],
                color='purple',
                label=f"Manifold = {manifold_distance:.2f}")

    ax.text(*point_a, "A", color='red')
    ax.text(*point_b, "B", color='green')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Linear Manifold Distance vs L2 Distance")

    plt.show()

# --- Main ---
if __name__ == "__main__":
    plot_s_shape()
    plot_linear_fold()
