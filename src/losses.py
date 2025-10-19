import torch
import ot
import numpy as np
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.sparse import csr_matrix

def get_cost_matrix(adj_matrix, num_nodes, method='dijkstra'):
    """
    Calculates the shortest path distance matrix.
    
    Args:
        adj_matrix: Sparse adjacency matrix (CSR format)
        num_nodes: Number of nodes in the graph
        method: 'dijkstra' (faster) or 'floyd_warshall' (exact but slower)
    
    Returns:
        torch.Tensor: Distance matrix
    """
    if method == 'dijkstra':
        dist_matrix = shortest_path(csgraph=adj_matrix, directed=False, unweighted=True)
    else:
        from scipy.sparse.csgraph import floyd_warshall
        dist_matrix = floyd_warshall(csgraph=adj_matrix, directed=False, unweighted=True)
    
    # Handle disconnected components
    dist_matrix[dist_matrix == float('inf')] = num_nodes
    dist_matrix[np.isnan(dist_matrix)] = num_nodes
    
    return torch.tensor(dist_matrix, dtype=torch.float)


def get_sampled_cost_matrix(adj_matrix, num_nodes, sample_size=500):
    """
    Calculates a sampled shortest path distance matrix for efficiency.
    
    For large graphs, computing the full cost matrix is expensive.
    This function samples a subset of nodes.
    
    Args:
        adj_matrix: Sparse adjacency matrix (CSR format)
        num_nodes: Number of nodes in the graph
        sample_size: Number of nodes to sample
    
    Returns:
        tuple: (sampled_cost_matrix, sampled_indices)
    """
    # Sample nodes
    sample_size = min(sample_size, num_nodes)
    sampled_indices = np.random.choice(num_nodes, size=sample_size, replace=False)
    
    # Compute shortest paths only for sampled nodes
    dist_matrix = np.zeros((sample_size, sample_size))
    
    for i, node_i in enumerate(sampled_indices):
        distances = dijkstra(csgraph=adj_matrix, directed=False, 
                           indices=node_i, unweighted=True)
        for j, node_j in enumerate(sampled_indices):
            dist_matrix[i, j] = distances[node_j]
    
    # Handle disconnected components
    dist_matrix[dist_matrix == float('inf')] = num_nodes
    dist_matrix[np.isnan(dist_matrix)] = num_nodes
    
    return torch.tensor(dist_matrix, dtype=torch.float), sampled_indices


def gromov_wasserstein_loss(C_source, C_target, p_source, p_target, 
                            max_iter=100, tol=1e-9):
    """
    Calculates the Gromov-Wasserstein loss between two graphs.
    
    Args:
        C_source (Tensor): Cost matrix (e.g., shortest paths) for the source graph.
        C_target (Tensor): Cost matrix for the target graph.
        p_source (Tensor): Node distribution for the source graph (usually uniform).
        p_target (Tensor): Node distribution for the target graph (usually uniform).
        max_iter (int): Maximum iterations for GW algorithm
        tol (float): Tolerance for convergence
    
    Returns:
        float: The GW loss.
    """
    # Convert to numpy for POT library
    C_s = C_source.cpu().numpy() if torch.is_tensor(C_source) else C_source
    C_t = C_target.cpu().numpy() if torch.is_tensor(C_target) else C_target
    p_s = p_source.cpu().numpy() if torch.is_tensor(p_source) else p_source
    p_t = p_target.cpu().numpy() if torch.is_tensor(p_target) else p_target
    
    # Ensure distributions sum to 1
    p_s = p_s / p_s.sum()
    p_t = p_t / p_t.sum()
    
    # Compute Gromov-Wasserstein distance
    try:
        loss = ot.gromov.gromov_wasserstein2(C_s, C_t, p_s, p_t, 
                                             loss_fun='square_loss',
                                             max_iter=max_iter,
                                             tol_rel=tol,
                                             tol_abs=tol,
                                             verbose=False)
        return torch.tensor(loss, dtype=torch.float)
    except Exception as e:
        print(f"Warning: GW computation failed: {e}")
        return torch.tensor(0.0, dtype=torch.float)


def sampled_gromov_wasserstein_loss(adj_source, adj_target, sample_size=500):
    """
    Computes Gromov-Wasserstein loss using sampled nodes for efficiency.
    
    Args:
        adj_source: Source graph adjacency matrix (CSR)
        adj_target: Target graph adjacency matrix (CSR)
        sample_size: Number of nodes to sample from each graph
    
    Returns:
        torch.Tensor: The sampled GW loss
    """
    num_source = adj_source.shape[0]
    num_target = adj_target.shape[0]
    
    # Get sampled cost matrices
    C_s, _ = get_sampled_cost_matrix(adj_source, num_source, sample_size)
    C_t, _ = get_sampled_cost_matrix(adj_target, num_target, sample_size)
    
    # Create uniform distributions
    p_s = torch.ones(C_s.shape[0]) / C_s.shape[0]
    p_t = torch.ones(C_t.shape[0]) / C_t.shape[0]
    
    return gromov_wasserstein_loss(C_s, C_t, p_s, p_t)