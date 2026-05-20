import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .utils import grad_reverse


class CausalEdgeMasker(nn.Module):
    """Learn a scalar causal weight in [0, 1] for each edge.

    For an edge (i, j) the weight is a function of the concatenated
    endpoint embeddings [h_i || h_j]. A sparsity penalty on these weights
    pushes the model toward keeping only the edges that are causally
    relevant for the label, encouraging an invariant subgraph across
    domains.
    """

    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        pair = torch.cat([h[src], h[dst]], dim=-1)
        return torch.sigmoid(self.mlp(pair)).squeeze(-1)


class CausalGNN_DANN(nn.Module):
    """Causal-masked GCN encoder + label predictor + domain discriminator.

    Pipeline (matches the notebook, adapted to the high-dim Twitch features):
      1. Project raw BoW features into a dense embedding.
      2. CausalEdgeMasker assigns a [0, 1] weight to every edge.
      3. Two GCNConv layers run message-passing weighted by those edges.
      4. Label predictor classifies nodes (binary `mature` flag).
      5. Domain classifier (GRL) predicts source vs target — gradient
         reversal makes the encoder learn domain-invariant features.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        num_domains: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, proj_dim)
        self.masker = CausalEdgeMasker(proj_dim, hidden_dim)

        self.conv1 = GCNConv(proj_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        h0 = F.relu(self.input_proj(x))
        edge_weights = self.masker(h0, edge_index)

        h = F.relu(self.conv1(h0, edge_index, edge_weights))
        h = self.conv2(h, edge_index, edge_weights)
        return h, edge_weights

    def forward(self, x, edge_index, alpha: float = 1.0):
        h, edge_weights = self.encode(x, edge_index)
        logits = self.label_predictor(h)
        domain_logits = self.domain_classifier(grad_reverse(h, alpha))
        return logits, domain_logits, edge_weights

    def predict_with_weights(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the label-prediction path with externally provided edge weights.

        Skips the masker so we can probe the model under counterfactual masks
        (e.g., 1 - w) for the causal counterfactual loss.
        """
        h0 = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h0, edge_index, edge_weights))
        h = self.conv2(h, edge_index, edge_weights)
        return self.label_predictor(h)
