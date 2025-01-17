import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
plt.style.use(['science', 'no-latex'])

# --- Create an 'S'-shaped manifold ---
def generate_s_shape(n_points=500, noise_std=0.01):
    """
    Generates an 'S' shape in 2D. 
    Adds slight Gaussian noise for realism.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = np.sin(t)
    y = 0.2 * np.sign(t - np.pi) * (1 - np.cos(t))  
    
    # Add slight Gaussian noise to 2D points
    noise = np.random.normal(0, noise_std, (n_points, 2))
    return np.column_stack([x, y]) + noise

def generate_linear_fold(n_points=484):
    """
    Generates a flat plane in 2D.
    By default, n_points=484 (22x22), so indices from 0 to 483 are valid.
    """
    side = int(np.round(np.sqrt(n_points)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, side),
        np.linspace(0, 1, side)
    )
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])

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
    query_idx = 250
    passage_idx = 20
    
    query = s_shape[query_idx]
    passage = s_shape[passage_idx]

    # Define radius for positive points
    radius = 1.2  # Adjust this value to change the size of the positive region
    
    # Calculate manifold distances from point A to all points
    distances_from_a = manifold_distances[query_idx]
    
    # Create boolean mask for points within radius
    positive_mask = distances_from_a <= radius

    # Compute Euclidean (L2) distance
    l2_distance = np.linalg.norm(query - passage)

    # Compute manifold (graph) distance
    manifold_distance = manifold_distances[query_idx, passage_idx]

    # Check if there is a valid path
    if np.isinf(manifold_distance):
        print(f"No path found between {query_idx} and {passage_idx}. Try increasing k.")
        path_coords = np.vstack([query, passage])
    else:
        path_indices = reconstruct_path(predecessors, query_idx, passage_idx)
        path_coords = s_shape[path_indices]

    # --- Plot ---
    plt.figure(figsize=(10, 8))

    # Scatter points with different colors based on distance
    plt.scatter(s_shape[~positive_mask, 0], s_shape[~positive_mask, 1],
               label="Negative Passages", color="grey", alpha=0.6)
    plt.scatter(s_shape[positive_mask, 0], s_shape[positive_mask, 1],
               label="Positive Passages", color="#FFBE7A", alpha=0.6)
    # Highlight chosen points
    plt.scatter(query[0], query[1], color='#FA7F6F', label="Query", marker='*', s=200, zorder=10)
    plt.scatter(passage[0], passage[1], color='#2878B5', label="Passage Found", marker='X', s=100, zorder=10)

    # Draw the straight (L2) line
    plt.plot([query[0], passage[0]],
             [query[1], passage[1]],
             linestyle='--', color='#D8383A',
             label=f"Euclidean Distance = {l2_distance:.2f}")

    # Draw manifold path if valid
    if len(path_coords) > 1:
        plt.plot(path_coords[:, 0],
                path_coords[:, 1],
                color='black',
                linestyle='--',
                label=f"Manifold Distance = {manifold_distance:.2f}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(frameon=True, framealpha=1, fancybox=False, edgecolor='black', facecolor='white')
    plt.title("Comparative Analysis of Manifold and Euclidean Distances in S-shaped Embedding Space Retrieval", fontsize=16)

    # save figure to pdf
    plt.savefig('s_shape.pdf', bbox_inches='tight')

# --- Plot linear fold ---
def plot_linear_fold():
    linear_fold = generate_linear_fold(n_points=484)

    # For a grid of 22x22, k=10 may or may not keep the whole graph connected.
    # Feel free to try k=15 or k=20 if you get no path.
    k = 8
    graph = kneighbors_graph(linear_fold, n_neighbors=k, mode='distance', include_self=False)
    # Compute all shortest paths
    manifold_distances, predecessors = shortest_path(graph, directed=False, return_predecessors=True)

    # Choose two points
    query_idx = 10
    passage_idx = 170  # valid because we have 0..483

    query = linear_fold[query_idx]
    passage = linear_fold[passage_idx]

    # Define radius for positive points
    radius = 0.5  # Adjust this value to change the size of the positive region
    
    # Calculate manifold distances from point A to all points
    distances_from_a = manifold_distances[query_idx]
    
    # Create boolean mask for points within radius
    positive_mask = distances_from_a <= radius

    # Euclidean distance
    l2_distance = np.linalg.norm(query - passage)

    # Manifold distance
    manifold_distance = manifold_distances[query_idx, passage_idx]

    if np.isinf(manifold_distance):
        print(f"No path found between {query_idx} and {passage_idx}. Try increasing k.")
        path_coords = np.vstack([query, passage])
    else:
        path_indices = reconstruct_path(predecessors, query_idx, passage_idx)
        path_coords = linear_fold[path_indices]

    # --- Plot ---
    plt.figure(figsize=(10, 8))

    # Scatter points with different colors based on distance
    plt.scatter(linear_fold[~positive_mask, 0], linear_fold[~positive_mask, 1],
               label="Negative Passages", color="#999999", alpha=0.6)
    plt.scatter(linear_fold[positive_mask, 0], linear_fold[positive_mask, 1],
               label="Positive Passages", color="#FFBE7A", alpha=0.6)

    # Points and direct L2 line
    plt.scatter(query[0], query[1], color='#FA7F6F', label="Query", marker='*', s=200, zorder=10)
    plt.scatter(passage[0], passage[1], color='#2878B5', label="Passage Found", marker='X', s=100, zorder=10)
    plt.plot([query[0], passage[0]],
             [query[1], passage[1]],
             linestyle='--', color='#D8383A',
             label=f"Euclidean Distance = {l2_distance:.2f}")

    # Manifold path
    if len(path_coords) > 1:
        plt.plot(path_coords[:, 0],
                path_coords[:, 1],
                linestyle='--',
                color='black',
                label=f"Manifold Distance = {manifold_distance:.2f}")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(frameon=True, framealpha=1, fancybox=False, edgecolor='black', facecolor='white')
    plt.title("Comparative Analysis of Manifold and Euclidean Distances in Linear Embedding Space Retrieval", fontsize=16)

    # save figure to pdf
    plt.savefig('linear_fold.pdf', bbox_inches='tight')

# --- Main ---
if __name__ == "__main__":
    plot_s_shape()
    plot_linear_fold()
