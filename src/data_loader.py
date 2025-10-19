import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import json
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def load_twitch_domain(data_path, domain_name, use_labels=True):
    """
    Loads features, adjacency matrix, and labels for a Twitch domain.
    
    Args:
        data_path (str): Path to the data folder.
        domain_name (str): The language domain to load (e.g., 'ENGB', 'FR').
        use_labels (bool): Whether to load labels (for source domain) or not (for target domain).
        
    Returns:
        tuple: (torch_geometric.data.Data, adjacency_matrix_csr)
            - Data object with features, edges, and optionally labels
            - Adjacency matrix in CSR format for OT loss computation
    """
    domain_path = os.path.join(data_path, domain_name)
    
    # Load edges
    edges_file = os.path.join(domain_path, f"musae_{domain_name}_edges.csv")
    edges_df = pd.read_csv(edges_file)
    edge_index = torch.tensor([edges_df['from'].values, edges_df['to'].values], dtype=torch.long)
    
    # Load target/node information
    target_file = os.path.join(domain_path, f"musae_{domain_name}_target.csv")
    target_df = pd.read_csv(target_file)
    
    # Create node ID mapping (original id -> new_id)
    id_mapping = dict(zip(target_df['id'], target_df['new_id']))
    num_nodes = len(target_df)
    
    # Load features from JSON
    features_file = os.path.join(domain_path, f"musae_{domain_name}_features.json")
    with open(features_file, 'r') as f:
        features_dict = json.load(f)
    
    # Determine feature dimension
    sample_key = list(features_dict.keys())[0]
    feature_dim = max(features_dict[sample_key]) + 1
    
    # Create feature matrix (one-hot encoding or bag-of-words)
    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    for node_id_str, feature_indices in features_dict.items():
        node_id = int(node_id_str)
        if node_id in id_mapping:
            new_id = id_mapping[node_id]
            features[new_id, feature_indices] = 1.0
    
    features = torch.tensor(features, dtype=torch.float)
    
    # Load labels if needed (use 'partner' column as binary label)
    labels = None
    if use_labels:
        labels = torch.tensor(target_df['mature'].astype(int).values, dtype=torch.long)
    
    # Create adjacency matrix for OT loss (CSR format)
    # Build adjacency matrix using new_ids
    edge_list = edges_df.values
    adj_data = np.ones(len(edge_list))
    
    # Map edges to new_id space
    row_indices = []
    col_indices = []
    for i, j in edge_list:
        if i in id_mapping and j in id_mapping:
            row_indices.append(id_mapping[i])
            col_indices.append(id_mapping[j])
    
    adj_matrix = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), 
                           shape=(num_nodes, num_nodes))
    # Make symmetric
    adj_matrix = adj_matrix + adj_matrix.T
    
    # Create PyG Data object
    data = Data(x=features, edge_index=edge_index, y=labels, num_nodes=num_nodes)
    
    return data, adj_matrix


def compute_adjacency_to_edge_index(adj_matrix, num_nodes):
    """
    Convert adjacency matrix to edge_index format.
    
    Args:
        adj_matrix: Sparse adjacency matrix
        num_nodes: Number of nodes
        
    Returns:
        torch.Tensor: Edge index in PyG format [2, num_edges]
    """
    adj_coo = adj_matrix.tocoo()
    edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
    return edge_index