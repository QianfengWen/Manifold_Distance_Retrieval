import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn import datasets, manifold
plt.style.use(['science', 'no-latex'])

# --- Create an 'S'-shaped manifold ---
def generate_s_shape(n_points=1500, noise_std=0.01):
    """
    Generates an swiss roll dataset
    """
    X, color = datasets.make_s_curve(n_samples=n_points, noise=noise_std, random_state=42)
    # Find points within the target region
    # Fixed the conditions - removed duplicate condition and adjusted ranges
    query_region = (X[:, 1] > 0.8) & (X[:, 1] < 1.2) & (X[:, 0] > 0.7) & (X[:, 0] < 1.3) & (X[:, 2] > 1)
    target_region = (X[:, 1] > -1.5) & (X[:, 1] < 1.5) & (X[:, 0] > -0.5) & (X[:, 0] < 0.5) & (X[:, 2] < 1) & (X[:, 2] > -1)
    
    # choose query and target
    query = X[query_region][np.argmax(X[query_region][:, 2])]
    print(query)
    target = X[target_region][np.argmin(X[target_region][:, 2])]
        
    return X, query, target

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
    s_shape, query, target = generate_s_shape(n_points=1500)

    # Use a reasonably large k to keep the graph connected
    k = 10
    graph = kneighbors_graph(s_shape, n_neighbors=k, mode='distance', include_self=False)
    # Compute all shortest paths
    manifold_distances, predecessors = shortest_path(graph, directed=False, return_predecessors=True)

    # Choose query point
    query_idx = np.where(s_shape == query)[0][0]
    target_idx = np.where(s_shape == target)[0][0]
    # Calculate manifold distances from query to all points
    manifold_distances_from_query = manifold_distances[query_idx]
    
    # calculate l2 distance from query to target
    l2_distances_from_query = np.linalg.norm(s_shape - query, axis=1)

    def create_plot(distances, title):
        fig = plt.figure(figsize=(7, 4), facecolor="white")
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize distances to [0,1] for better color contrast
        distances_normalized = distances / distances.max()
        
        # Create mask for non-query and non-target points
        mask = ~((s_shape[:, 0] == query[0]) & (s_shape[:, 1] == query[1]) & (s_shape[:, 2] == query[2])) & ~((s_shape[:, 0] == target[0]) & (s_shape[:, 1] == target[1]) & (s_shape[:, 2] == target[2]))
        points_to_plot = s_shape[mask]
        distances_to_plot = distances_normalized[mask]
        
        col = ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], points_to_plot[:, 2],
                        c= distances_to_plot,
                        cmap='YlGnBu',
                        s=5,
                        alpha=0.7)
        
        # Plot query point
        
        ax.plot(query[0], query[1], query[2], 
                color='#934B43',
                marker='*',
                linewidth=3,
                zorder=200)
        # include label 'Query' as a point in legend
        ax.scatter([], [], [], label='Query', color='#934B43', marker='*', s=50, zorder=200)
        
        # Plot target point
        ax.plot(target[0], target[1], target[2],
                color='#14517C',
                marker='*',
                linewidth=3,
                zorder=200)
        # include label 'Target' as a point in legend
        ax.scatter([], [], [], label='Sample Passage', color='#14517C', marker='*', s=50, zorder=200)
        
        # Draw L2 (straight) path
        ax.plot([query[0], target[0]], 
                [query[1], target[1]], 
                [query[2], target[2]], 
                'r--', 
                label='Euclidean Path',
                color='#934B43',
                linewidth=2,
                zorder=100)
        
        # Draw manifold path if this is the manifold distance plot
        if title == 'Manifold Distances':
            path_indices = reconstruct_path(predecessors, query_idx, target_idx)
            path_points = s_shape[path_indices]
            ax.plot(path_points[:, 0], 
                    path_points[:, 1], 
                    path_points[:, 2], 
                    'g--', 
                    label='Manifold Path',
                    color='#14517C',
                    linewidth=2,
                    zorder=100)
        
        ax.view_init(azim=-60, elev=9)
        # remove ticks lebel, but keep grid
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.legend(fontsize=10, loc = 'lower center', ncol=4)
        
        cb = fig.colorbar(col, orientation="vertical", 
                shrink=0.7, aspect=20, pad=0.1)
        cb.ax.invert_yaxis()
        cb.ax.set_title('Relevance',fontsize=10)
        cb.ax.set_xticks([])  # Remove x ticks
        cb.ax.set_yticks([])  # Remove y ticks

        plt.tight_layout()
        return fig

    # Create and save manifold distances plot
    fig1 = create_plot(manifold_distances_from_query, 'Manifold Distances')
    fig1.savefig('manifold_distances.pdf', bbox_inches='tight')
    
    plt.show()

# --- Main ---
if __name__ == "__main__":
    plot_s_shape()
